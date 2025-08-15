from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import random

from mesa import Agent
from .constants import RESOURCES, WOOD, LIVESTOCK, STONE
from .utils import clamp

@dataclass
class Traits:
    # Gathering efficiency [0, 10] per resource
    wood: float
    livestock: float
    stone: float

    def as_dict(self) -> Dict[str, float]:
        return {WOOD: self.wood, LIVESTOCK: self.livestock, STONE: self.stone}

    @staticmethod
    def random():
        return Traits(
            wood=random.uniform(0, 10),
            livestock=random.uniform(0, 10),
            stone=random.uniform(0, 10),
        )

    @staticmethod
    def from_parents(a: 'Traits', b: 'Traits', sigma: float = 0.75) -> 'Traits':
        # simple blend crossover + gaussian mutation
        def blend(x, y):
            base = 0.5 * (x + y)
            return clamp(random.gauss(base, sigma), 0.0, 10.0)

        return Traits(
            wood=blend(a.wood, b.wood),
            livestock=blend(a.livestock, b.livestock),
            stone=blend(a.stone, b.stone),
        )

class GathererAgent(Agent):
    """An agent that gathers one type of resource per day based on skill and community need."""
    def __init__(self, unique_id, model, community_id: int, traits: Traits):
        super().__init__(unique_id, model)
        self.community_id = community_id
        self.traits = traits

        # Tracking
        self.daily_choice_history = []  # resource chosen each day
        self.monthly_contribution = {r: 0.0 for r in RESOURCES}

    # --- Decision policy ---
    def choose_resource(self) -> str:
        community = self.model.communities[self.community_id]
        # Need score: higher if stock is far below threshold
        need = {}
        for r in RESOURCES:
            threshold_r = community.thresholds[r]
            stock_r = community.stocks[r]
            deficit_ratio = max(0.0, (threshold_r - stock_r) / max(1.0, threshold_r))
            # Mix: community's bartering weights + real-time deficit
            need[r] = 0.5 * community.barter_weights[r] + 0.5 * deficit_ratio

        # Skill-modulated utility
        utilities = {}
        for r in RESOURCES:
            skill = getattr(self.traits, r)
            availability = community.local_supply[r]  # if 0, disincentivize
            utilities[r] = (skill / 10.0) * (need[r]) * (1.0 if availability > 1e-6 else 0.1)

        # Soft choice with mild exploration
        total = sum(utilities.values())
        if total <= 0:
            return random.choice(RESOURCES)
        probs = {r: (u / total) for r, u in utilities.items()}
        # epsilon-greedy
        eps = 0.05
        if random.random() < eps:
            return random.choice(RESOURCES)
        # pick max-prob
        return max(probs, key=probs.get)

    def step(self):
        community = self.model.communities[self.community_id]
        resource = self.choose_resource()
        self.daily_choice_history.append(resource)

        # Gather amount based on skill and capped by local supply
        skill = getattr(self.traits, resource)  # 0-10
        base_rate = self.model.config.base_gather_rate  # units per day at skill=10
        amount = base_rate * (skill / 10.0)
        # Small noise
        amount *= random.uniform(0.9, 1.1)
        gathered = community.consume_local_supply(resource, amount)
        # Add to community stocks
        community.stocks[resource] += gathered
        self.monthly_contribution[resource] += gathered

    def reset_monthly_tracking(self):
        self.daily_choice_history.clear()
        self.monthly_contribution = {r: 0.0 for r in RESOURCES}