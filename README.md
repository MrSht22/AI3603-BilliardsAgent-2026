# BilliardsAgent: Intelligent Billiards Agent

## Project Overview

This project implements an intelligent agent for competitive 8-ball billiards, developed as the final project for course AI3603 (Reinforcement Learning). The agent competes against a baseline opponent in a physics-based billiards simulation environment (`pooltool`).

The objective is to design a decision-making agent that outperforms the baseline `BasicAgent` through physics-based shot calculation and noise-robust action evaluation.

---

## Project Structure

```
BilliardsAgent-2026/
├── agents/                    # Agent implementations
│   ├── agent.py              # Abstract Agent base class
│   ├── basic_agent.py        # Baseline Bayesian optimization agent
│   ├── basic_agent_pro.py    # Enhanced baseline agent
│   └── new_agent.py          # Our proposed agent (main entry)
├── train/                     # Training scripts and documentation
│   ├── train_sac.py          # SAC reinforcement learning (exploratory)
│   └── README.md             # Training documentation
├── eval/                      # Evaluation scripts and documentation
│   ├── evaluate.py           # Main evaluation script
│   ├── new_agent.py          # Final agent wrapper for evaluation
│   └── README.md             # Evaluation documentation
├── poolenv.py                # Billiards environment (DO NOT MODIFY)
├── GAME_RULES.md             # Detailed game rules
├── PROJECT_GUIDE.md          # Project requirements and guidelines
└── report.pdf                # Final project report
```

---

## Game Rules

The environment follows standard **8-Ball Billiards** rules:

- **16 balls**: 1 cue ball, 7 solid (1-7), 7 stripe (9-15), 1 eight-ball (8)
- **Objective**: Clear your assigned ball type, then legally pocket the 8-ball
- **Fouls**: Cue ball pocketed, wrong first hit, no cushion contact — result in turn transfer
- **Instant loss**: Pocketing 8-ball prematurely or simultaneous cue+8-ball
- **Timeout**: 60 shots per player triggers remaining-ball count comparison

Detailed rules are specified in `GAME_RULES.md`.

---

## Our Approach

### Algorithm Design

The proposed `NewAgent` combines **physics-based shot planning** with **noise-aware Monte Carlo evaluation**:

1. **Candidate Generation**: For each target ball and pocket combination, compute geometric shot parameters (aim point, angle, velocity) using collision physics
2. **Variant Expansion**: Generate multiple action variants around each candidate (adjusting velocity and angle) to hedge against execution noise
3. **Monte Carlo Simulation**: Evaluate each variant's expected score by simulating multiple noisy executions
4. **Safety Policy**: Fall back to a defensive shot when no viable offensive option exists

### Key Features

| Feature | Description |
|---------|-------------|
| Path occlusion detection | Rejects shots blocked by other balls |
| Cut angle optimization | Penalizes high-difficulty angle shots |
| Noise-robust evaluation | Uses multiple simulations to estimate expected payoff |
| Configurable strategies | Three presets (BASELINE, AGGRESSIVE, CONSERVATIVE) |

### Exploratory Approaches

Prior to final implementation, the following methods were investigated:

- **SAC Reinforcement Learning**: Attempted but ultimately not adopted due to training instability and suboptimal sample efficiency in this episodic, physics-heavy domain
- **Bayesian Optimization**: Used in the baseline agent for parameter tuning

---

## Environment Setup

### Requirements

- **OS**: Ubuntu 22.04 (recommended)
- **Python**: 3.13
- **CUDA**: Optional (for faster training)

### Installation

```bash
# Create conda environment
conda create -n poolenv python=3.13
conda activate poolenv

# Clone pooltool
git clone https://github.com/SJTU-RL2/pooltool.git
cd pooltool

# Install poetry and dependencies
pip install "poetry==2.2.1"
poetry install --with=dev,docs

# Install additional dependencies for baseline agent
pip install bayesian-optimization numpy
```

---

## Usage

### Training (Exploratory)

```bash
conda activate poolenv
cd train
python train_sac.py
```

### Evaluation

```bash
conda activate poolenv
cd eval
python evaluate.py
```

The evaluation runs **120 games** (4-agent fairness rotation) and reports:
- Win/loss/draw counts per agent
- Normalized score (win + 0.5 × draw)
- Win rate against baseline

---

## Configuration

### Agent Configuration

`NewAgent` supports configurable strategies via constructor parameters:

```python
from agents.new_agent import NewAgent

agent = NewAgent(
    variant_config='BASELINE',    # Variant generation: 'BASELINE', 'AGGRESSIVE', 'CONSERVATIVE'
    scoring_config='BASELINE',    # Scoring weights: 'BASELINE', 'HIGH_REWARD_LOW_PENALTY', 'LOW_REWARD_HIGH_PENALTY'
    n_simulations=8               # Monte Carlo simulations per candidate
)
```

### Noise Parameters

Execution noise is injected into all shot parameters to simulate real-world variance:

| Parameter | Description | Std Dev |
|-----------|-------------|---------|
| V0 | Initial velocity (m/s) | 0.1 |
| phi | Horizontal angle (deg) | 0.15 |
| theta | Vertical angle (deg) | 0.1 |
| a | Lateral cue offset | 0.005 |
| b | Longitudinal cue offset | 0.005 |

---

## Results

Performance evaluation against `BasicAgent` over 120 games:

| Metric | Value |
|--------|-------|
| Win Rate | > 60% |
| Score (out of 120) | > 72 |
| vs. BasicAgentPro | > 88% (recommended) |

See `report.pdf` for detailed experimental results and analysis.

---

## Documentation

- `GAME_RULES.md` — Complete game rules and evaluation protocol
- `PROJECT_GUIDE.md` — Assignment requirements and submission guidelines
- `train/README.md` — Training configuration and hyperparameters
- `eval/README.md` — Evaluation usage and development history

---

## References

- Pooltool: [https://github.com/SJTU-RL2/pooltool](https://github.com/SJTU-RL2/pooltool)
- IEEE Conference Template: [Overleaf](https://www.overleaf.com/1998687845fyyfzhmpnfkd#b117f2)
