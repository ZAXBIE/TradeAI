# Cooperative Trade Among AI Communities (Mesa Simulation)

This project simulates cooperative trade among three resource‑specialized communities using an agent‑based model (ABM) built with **Mesa**. 
Communities must barter to obtain missing resources. Agents have gathering traits that evolve via a simple genetic algorithm. 
Bartering weights adapt over generations based on success/failure at meeting survival thresholds.


---

## Features

- Three communities with asymmetric resource endowments: **wood**, **livestock**, **stone**
- Agents with per‑resource gathering skills (0–10)
- Daily gathering guided by skills and community needs
- Monthly (generation) **barter** between communities using evolving **bartering weights**
- **Genetic algorithm**: top two contributors to the lacking resource reproduce to create one mutated offspring per generation
- Environmental **resource supply** with regeneration (abundant/ scarce/ none)
- Survival thresholds and **population loss** proportional to deficits
- Data logged to CSV; quick **Matplotlib** plots included

---
## PYCHARM Setup

Unzip & open the project

In PyCharm: File → Open… → select the community_trade_sim_v2 folder → Open as Project.

Mark src as a Sources Root (important)

In the Project tool window, right‑click the src folder → Mark Directory as → Sources Root.
(This ensures imports like from src.community_trade... work.)

Create a virtual environment

File → Settings → Project: … → Python Interpreter.

Click the gear icon → Add… → New environment (Virtualenv). Pick your base interpreter (e.g., Python 3.10+).

Click OK.

Install dependencies

PyCharm usually detects requirements.txt and offers to install. If not:

Open Terminal (bottom) and run:

pip install -r requirements.txt


Create a Run/Debug configuration

Run → Edit Configurations… → + → Python.

Name: run_experiment

Script path: select run_experiment.py in the project root.

Working directory: set to the project root (community_trade_sim_v2).
(Critical for the src/... imports.)

Parameters (optional):

--generations 24 --seed 42


Python interpreter: choose the venv you created.

Apply → OK.

Run

Click the green ▶️ or Run → Run 'run_experiment'.

On completion you’ll see:

outputs/run_YYYYmmdd_HHMMSS/results.csv

outputs/run_YYYYmmdd_HHMMSS/plots/*.png


## For Linux Terminal Install
## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Mesa is required but the code also runs headless (no browser UI) for experiments.

---

## Run a quick experiment

```bash
python run_experiment.py --generations 24 --seed 42
```

Outputs:
- `outputs/run_YYYYmmdd_HHMMSS/results.csv` – per‑generation metrics
- `outputs/run_YYYYmmdd_HHMMSS/plots/` – PNG charts

To re‑plot later:
```bash
python plot_results.py outputs/run_YYYYmmdd_HHMMSS/results.csv
```

---

## Model Overview

- **Tick = day**, **30 days = 1 generation (month)**.
- Each day, agents choose a resource to gather using their skill and community need. 
- Each end‑of‑month:
  1. Pairwise barter between communities using exchange rates derived from bartering weights (community‑level).  
     Prices are the mean of the two trading communities' weights for each resource.
  2. Evaluate community deficits vs thresholds → **mortality** (remove weakest contributors).
  3. **Reproduction**: top two contributors to the most lacking resource produce one mutated agent (added, not replaced).
  4. Small adaptive update and mutation to bartering weights (learn from deficits).
- Wood, livestock, and stone each have **local supplies** per community with caps and daily regeneration, from which agents gather.

---

## Files

```
community_trade_sim/
├── requirements.txt
├── README.md
├── run_experiment.py
├── plot_results.py
└── src/
    └── community_trade/
        ├── __init__.py
        ├── constants.py
        ├── utils.py
        ├── agents.py
        ├── trade.py
        └── model.py
```

---

## Citations (Background & Related Work)

- Wikimedia Foundation. (2023, December 24). Sugarscape. Wikipedia. https://en.wikipedia.org/wiki/Sugarscape
- Growing agent-based artificial societies. Sugarscape. (n.d.). https://sugarscape.sourceforge.net/
- NetLogo models library. (n.d.). https://ccl.northwestern.edu/netlogo/models/index.cgi
```