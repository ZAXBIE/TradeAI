from __future__ import annotations
from typing import Dict, Tuple
from .constants import RESOURCES
from .utils import add_dicts, clip_nonnegative

def average_price(a_weights: Dict[str, float], b_weights: Dict[str, float]) -> Dict[str, float]:
    return {r: 0.5 * (a_weights[r] + b_weights[r]) for r in RESOURCES}

def compute_surplus_deficit(stocks: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    surplus = {}
    deficit = {}
    for r in RESOURCES:
        s = stocks[r] - thresholds[r]
        if s > 0:
            surplus[r] = s
            deficit[r] = 0.0
        else:
            surplus[r] = 0.0
            deficit[r] = -s
    return surplus, deficit

def trade_pair(a, b) -> Dict[str, float]:
    """Execute barter between community a and b.
    Returns trade volumes aggregated by resource (positive means net inflow to 'a', negative outflow).
    Symmetric opposite will be applied to 'b' by caller.
    """
    # Compute pre-trade surplus/deficit
    a_surplus, a_deficit = compute_surplus_deficit(a.stocks, a.thresholds)
    b_surplus, b_deficit = compute_surplus_deficit(b.stocks, b.thresholds)

    prices = average_price(a.barter_weights, b.barter_weights)

    trade_log_a = {r: 0.0 for r in RESOURCES}

    progress = True
    while progress:
        progress = False
        # Find a needed resource ra that b can supply, and a resource rb that a can give which b needs
        ra_candidates = [r for r in RESOURCES if a_deficit[r] > 1e-9 and b_surplus[r] > 1e-9]
        rb_candidates = [r for r in RESOURCES if b_deficit[r] > 1e-9 and a_surplus[r] > 1e-9]
        if not ra_candidates or not rb_candidates:
            break

        # Greedy pair: pick the most valuable to A and B respectively
        # Value by their own weights scaled by deficit
        ra = max(ra_candidates, key=lambda r: a.barter_weights[r] * a_deficit[r])
        rb = max(rb_candidates, key=lambda r: b.barter_weights[r] * b_deficit[r])

        p_ra = prices[ra]
        p_rb = prices[rb]

        # Max feasible amount of ra (from b to a)
        x_cap_b = b_surplus[ra]
        # Limited by a's ability to pay in rb: y <= a_surplus[rb]; and y = x * p_ra / p_rb
        x_cap_a_pay = a_surplus[rb] * (p_rb / p_ra) if p_ra > 0 else 0.0
        # Also don't exceed a's need and b's need
        x_cap_a_need = a_deficit[ra]
        x_cap_b_need = b_deficit[rb] * (p_rb / p_ra) if p_ra > 0 else 0.0

        x = min(x_cap_b, x_cap_a_pay, x_cap_a_need, x_cap_b_need)
        if x <= 1e-9:
            break

        y = x * (p_ra / p_rb)  # amount of rb from a to b

        # Apply the trade
        a.stocks[ra] += x
        a.stocks[rb] -= y
        b.stocks[ra] -= x
        b.stocks[rb] += y

        a_surplus, a_deficit = compute_surplus_deficit(a.stocks, a.thresholds)
        b_surplus, b_deficit = compute_surplus_deficit(b.stocks, b.thresholds)

        trade_log_a[ra] += x
        trade_log_a[rb] -= y

        progress = True

    return trade_log_a