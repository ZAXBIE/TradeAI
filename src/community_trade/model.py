from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import random
import time

import numpy as np
import pandas as pd
from mesa import Model
from mesa.time import RandomActivation

from .constants import RESOURCES, WOOD, LIVESTOCK, STONE, DAYS_PER_MONTH
from .agents import GathererAgent, Traits
from .trade import trade_pair, compute_surplus_deficit
from .utils import normalize_weights, zeros_like, dict_min_key, clamp, dirichlet_perturb

# ----------------------------- Configs -----------------------------

@dataclass
class CommunityConfig:
    name: str
    id: int
    initial_population: int = 10

    # Local resource supply caps
    cap_abundant: float = 1000.0
    cap_scarce: float = 300.0
    cap_none: float = 0.0

    # Daily regeneration amounts
    regen_abundant: float = 80.0
    regen_scarce: float = 15.0
    regen_none: float = 0.0

    # Map resource -> which endowment type for this community
    endowment: Dict[str, str] = field(default_factory=lambda: {
        WOOD: 'abundant', LIVESTOCK: 'none', STONE: 'scarce'
    })

    # Monthly thresholds required to be considered "thriving"
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        WOOD: 400.0, LIVESTOCK: 400.0, STONE: 400.0
    })

    # Initial bartering weights (relative value importance; will be normalized)
    bartering_weights: Dict[str, float] = field(default_factory=lambda: {
        WOOD: 1.0, LIVESTOCK: 1.0, STONE: 1.0
    })

@dataclass
class SimulationConfig:
    generations: int = 24
    days_per_month: int = DAYS_PER_MONTH
    base_gather_rate: float = 10.0  # units/day at skill=10
    seed: int | None = None

    # Mortality
    mortality_scale: float = 0.5  # remove up to 50% of population when fully deficit across all resources

    # Reproduction
    offspring_per_generation: int = 1
    trait_mutation_sigma: float = 0.75

    # Weights learning
    weight_learning_rate: float = 0.25
    weight_dirichlet_alpha_scale: float = 30.0  # lower -> higher noise

# ----------------------------- Community state -----------------------------

class CommunityState:
    def __init__(self, config: CommunityConfig):
        self.config = config
        self.id = config.id
        self.name = config.name

        # stocks: accumulated resources held by the community
        self.stocks: Dict[str, float] = {r: 0.0 for r in RESOURCES}

        # local supply: available to gather today (regenerates daily up to a cap)
        self.local_supply: Dict[str, float] = {}
        self.local_caps: Dict[str, float] = {}
        self.local_regen: Dict[str, float] = {}

        for r in RESOURCES:
            endow = config.endowment[r]
            cap = getattr(config, f"cap_{endow}")
            regen = getattr(config, f"regen_{endow}")
            self.local_supply[r] = cap * 0.5  # start at half capacity
            self.local_caps[r] = cap
            self.local_regen[r] = regen

        # thresholds per month
        self.thresholds = dict(config.thresholds)

        # bartering weights normalized
        self.barter_weights = normalize_weights(dict(config.bartering_weights))

        # population agents list will be managed by Mesa scheduler
        self.agent_ids: List[int] = []

        # trade tracking within a generation
        self.trade_log = {r: 0.0 for r in RESOURCES}
        self.births = 0
        self.deaths = 0

    # --- mechanics ---
    def consume_local_supply(self, resource: str, amount: float) -> float:
        take = min(self.local_supply[resource], amount)
        self.local_supply[resource] -= take
        return take

    def regenerate_local_supply(self):
        for r in RESOURCES:
            self.local_supply[r] = min(self.local_supply[r] + self.local_regen[r], self.local_caps[r])

    def lacking_resource(self) -> str:
        # Choose the resource with lowest stock/threshold ratio
        ratios = {r: (self.stocks[r] / max(1.0, self.thresholds[r])) for r in RESOURCES}
        return min(ratios, key=ratios.get)

    # Learning: adjust weights toward needs; then inject small Dirichlet perturbation
    def update_barter_weights(self, learning_rate: float, alpha_scale: float):
        _, deficits = compute_surplus_deficit(self.stocks, self.thresholds)
        # Move weights toward deficit proportions
        total_def = sum(deficits.values())
        if total_def > 0:
            desired = {r: deficits[r] / total_def for r in RESOURCES}
            for r in RESOURCES:
                self.barter_weights[r] = (1 - learning_rate) * self.barter_weights[r] + learning_rate * desired[r]
        # Regularize via Dirichlet noise (keeps sum=1, positivity)
        self.barter_weights = normalize_weights(dirichlet_perturb(self.barter_weights, alpha_scale=alpha_scale))

# ----------------------------- Model -----------------------------

class TradeModel(Model):
    def __init__(self, community_configs: List[CommunityConfig], config: SimulationConfig):
        super().__init__()
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        self.schedule = RandomActivation(self)
        self.communities: Dict[int, CommunityState] = {}
        self.agent_counter = 0
        self.generation = 0

        # Create communities
        for cc in community_configs:
            c = CommunityState(cc)
            self.communities[c.id] = c
            # Spawn agents
            for _ in range(cc.initial_population):
                self._spawn_agent(c)

    # ---------------- helpers ----------------

    def _spawn_agent(self, community: CommunityState, traits: Traits | None = None):
        if traits is None:
            traits = Traits.random()
        self.agent_counter += 1
        a = GathererAgent(self.agent_counter, self, community.id, traits)
        self.schedule.add(a)
        community.agent_ids.append(a.unique_id)

    def _remove_agent(self, unique_id: int, community: CommunityState):
        agent = next((ag for ag in self.schedule.agents if ag.unique_id == unique_id), None)
        if agent is not None:
            self.schedule.remove(agent)
        if unique_id in community.agent_ids:
            community.agent_ids.remove(unique_id)

    # ---------------- daily loop ----------------

    def step_day(self):
        # Agents gather
        self.schedule.step()
        # Regenerate supplies
        for c in self.communities.values():
            c.regenerate_local_supply()

    # ---------------- monthly logic ----------------

    def _pairwise_trade(self):
        ids = list(self.communities.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.communities[ids[i]]
                b = self.communities[ids[j]]
                delta_a = trade_pair(a, b)
                # track both sides
                for r in RESOURCES:
                    a.trade_log[r] += delta_a[r]
                    b.trade_log[r] -= delta_a[r]

    def _evaluate_and_demography(self):
        """Apply survival evaluation and reproduction per community."""
        for c in self.communities.values():
            # Mortality proportional to total normalized deficit
            _, deficit = compute_surplus_deficit(c.stocks, c.thresholds)
            deficit_ratio = sum(deficit.values()) / max(1.0, sum(c.thresholds.values()))
            pop = len(c.agent_ids)
            to_remove = int(math.floor(self.config.mortality_scale * deficit_ratio * pop))
            if to_remove > 0 and pop > 0:
                # Remove the worst contributors
                agents = [ag for ag in self.schedule.agents if isinstance(ag, GathererAgent) and ag.community_id == c.id]
                # Score by contribution to lacking resource
                lacking = c.lacking_resource()
                ranked = sorted(agents, key=lambda ag: ag.monthly_contribution.get(lacking, 0.0))
                for ag in ranked[:to_remove]:
                    self._remove_agent(ag.unique_id, c)
                c.deaths += to_remove

            # Reproduction: pick top two contributors to lacking resource
            agents = [ag for ag in self.schedule.agents if isinstance(ag, GathererAgent) and ag.community_id == c.id]
            if len(agents) >= 2:
                lacking = c.lacking_resource()
                ranked = sorted(agents, key=lambda ag: ag.monthly_contribution.get(lacking, 0.0), reverse=True)
                p1, p2 = ranked[0], ranked[1]
                child_traits = Traits.from_parents(p1.traits, p2.traits, sigma=self.config.trait_mutation_sigma)
                for _ in range(self.config.offspring_per_generation):
                    self._spawn_agent(c, child_traits)
                    c.births += 1

            # Weights learning
            c.update_barter_weights(self.config.weight_learning_rate, self.config.weight_dirichlet_alpha_scale)

            # Reset agent monthly tracking
            for ag in self.schedule.agents:
                if isinstance(ag, GathererAgent) and ag.community_id == c.id:
                    ag.reset_monthly_tracking()

    # ---------------- public API ----------------

    def run(self, generations: int | None = None) -> pd.DataFrame:
        if generations is None:
            generations = self.config.generations
        records = []
        for g in range(generations):
            self.generation = g + 1
            # Run days
            for _ in range(self.config.days_per_month):
                self.step_day()
            # Trade
            self._pairwise_trade()
            # Demography and learning
            self._evaluate_and_demography()
            # Log
            ts = time.time()
            for c in self.communities.values():
                surplus, deficit = compute_surplus_deficit(c.stocks, c.thresholds)
                rec = {
                    "generation": self.generation,
                    "timestamp": ts,
                    "community_id": c.id,
                    "community_name": c.name,
                    "population": len(c.agent_ids),
                    "stock_wood": c.stocks[WOOD],
                    "stock_livestock": c.stocks[LIVESTOCK],
                    "stock_stone": c.stocks[STONE],
                    "deficit_wood": deficit[WOOD],
                    "deficit_livestock": deficit[LIVESTOCK],
                    "deficit_stone": deficit[STONE],
                    "trade_wood": c.trade_log[WOOD],
                    "trade_livestock": c.trade_log[LIVESTOCK],
                    "trade_stone": c.trade_log[STONE],
                    "weight_wood": c.barter_weights[WOOD],
                    "weight_livestock": c.barter_weights[LIVESTOCK],
                    "weight_stone": c.barter_weights[STONE],
                    "births": c.births,
                    "deaths": c.deaths,
                }
                records.append(rec)
                # Reset trade counters for next month
                c.trade_log = {r: 0.0 for r in RESOURCES}
                c.births = 0
                c.deaths = 0
        return pd.DataFrame.from_records(records)

# ----------------------------- Factory -----------------------------

def default_communities() -> List[CommunityConfig]:
    # Community 1: abundant wood, scarce stone, no livestock
    c1 = CommunityConfig(
        name="Woodland",
        id=1,
        endowment={WOOD: 'abundant', LIVESTOCK: 'none', STONE: 'scarce'},
        bartering_weights={WOOD: 1.0, LIVESTOCK: 1.0, STONE: 1.0},
    )
    # Community 2: abundant livestock, scarce wood, no stone
    c2 = CommunityConfig(
        name="Pasture",
        id=2,
        endowment={WOOD: 'scarce', LIVESTOCK: 'abundant', STONE: 'none'},
        bartering_weights={WOOD: 1.0, LIVESTOCK: 1.0, STONE: 1.0},
    )
# Community 3: abundant stone, scarce livestock, no wood
    c3 = CommunityConfig(
        name="Quarry",
        id=3,
        endowment={WOOD: 'none', LIVESTOCK: 'scarce', STONE: 'abundant'},
        bartering_weights={WOOD: 1.0, LIVESTOCK: 1.0, STONE: 1.0},
    )
    return [c1, c2, c3]