from __future__ import annotations
import math
import random
from typing import Dict, Iterable, List
import numpy as np

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        # fallback to uniform
        n = len(w)
        return {k: 1.0 / n for k in w.keys()}
    return {k: max(0.0, v) / s for k, v in w.items()}

def softmax_dict(d: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    xs = np.array(list(d.values()), dtype=float) / max(1e-9, temperature)
    xs = xs - xs.max()
    exps = np.exp(xs)
    exps_sum = exps.sum()
    probs = exps / (exps_sum if exps_sum > 0 else 1.0)
    return {k: float(p) for k, p in zip(d.keys(), probs)}

def weighted_choice(probs: Dict[str, float]) -> str:
    r = random.random()
    cumulative = 0.0
    for k, p in probs.items():
        cumulative += p
        if r <= cumulative:
            return k
    # numerical fallback
    return list(probs.keys())[-1]

def dict_min_key(d: Dict[str, float]) -> str:
    return min(d, key=d.get)

def dict_max_key(d: Dict[str, float]) -> str:
    return max(d, key=d.get)

def add_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in set(a) | set(b)}

def sub_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}

def zeros_like(keys: Iterable[str]) -> Dict[str, float]:
    return {k: 0.0 for k in keys}

def clip_nonnegative(d: Dict[str, float]) -> Dict[str, float]:
    return {k: max(0.0, v) for k, v in d.items()}

def as_tuple(d: Dict[str, float], order: List[str]) -> tuple:
    return tuple(d[o] for o in order)

def gaussian_mutation(value: float, sigma: float, lo: float, hi: float) -> float:
    return clamp(value + random.gauss(0.0, sigma), lo, hi)

def dirichlet_perturb(weights: Dict[str, float], alpha_scale: float = 50.0) -> Dict[str, float]:
    # Convert to Dirichlet params roughly proportional to existing weights
    keys = list(weights.keys())
    base = np.array([max(1e-6, weights[k]) for k in keys], dtype=float)
    alpha = alpha_scale * base / base.sum()
    sample = np.random.dirichlet(alpha)
    return {k: float(v) for k, v in zip(keys, sample)}