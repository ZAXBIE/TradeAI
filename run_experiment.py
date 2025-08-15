from __future__ import annotations
import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from src.community_trade.model import TradeModel, SimulationConfig, default_communities
from src.community_trade.constants import WOOD, LIVESTOCK, STONE

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run cooperative trade simulation")
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--days-per-month", type=int, default=30)
    args = parser.parse_args()

    out_dir = os.path.join("outputs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    config = SimulationConfig(
        generations=args.generations,
        days_per_month=args.days_per_month,
        seed=args.seed,
    )
    model = TradeModel(default_communities(), config)
    df = model.run()

    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    # Basic plots
    def plot_metric(df, y, title, fname):
        import matplotlib.pyplot as plt
        plt.figure()
        markers = ['o', 's', '^']  # different point shapes
        for (i, (cid, g)) in enumerate(sorted(df.groupby("community_id"))):
            plt.plot(g["generation"], g[y], marker=markers[i % len(markers)], label=f"Community {cid}")
        plt.xlabel("Generation (month)")
        plt.ylabel(y.replace("_", " ").title())
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
        plt.close()

    plot_metric(df, "population", "Population Over Time", "population.png")
    plot_metric(df, "stock_wood", "Stock: Wood", "stock_wood.png")
    plot_metric(df, "stock_livestock", "Stock: Livestock", "stock_livestock.png")
    plot_metric(df, "stock_stone", "Stock: Stone", "stock_stone.png")

    for y in ["weight_wood", "weight_livestock", "weight_stone"]:
        plot_metric(df, y, f"Bartering Weight: {y.split('_')[1].title()}", f"{y}.png")

    print("Run complete.")
    print("Results CSV:", csv_path)
    print("Plots in:", plots_dir)

if __name__ == "__main__":
    main()