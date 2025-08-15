from __future__ import annotations
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot results from results.csv")
    parser.add_argument("csv_path", type=str, help="Path to results.csv")
    parser.add_argument("--outdir", type=str, default=None, help="Directory for plots (defaults next to CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    outdir = args.outdir or os.path.join(os.path.dirname(args.csv_path), "plots_cli")
    os.makedirs(outdir, exist_ok=True)

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
    for y in ["stock_wood", "stock_livestock", "stock_stone",
              "deficit_wood", "deficit_livestock", "deficit_stone",
              "trade_wood", "trade_livestock", "trade_stone",
              "weight_wood", "weight_livestock", "weight_stone"]:
        plot_metric(df, y, y.replace("_", " ").title(), f"{y}.png")

    print("Wrote plots to", outdir)

if __name__ == "__main__":
    main()