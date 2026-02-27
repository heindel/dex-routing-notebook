# Gas-Aware DEX Routing — Interactive Pluto Notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://binder.plutojl.org/open?url=https%3A%2F%2Fraw.githubusercontent.com%2Fheindel%2Fdex-routing-notebook%2Fmain%2Fdex_routing.jl)

**Click the badge above to launch the interactive notebook in your browser — no installation required.**

Companion notebook to *"Finding Solutions on the Ethereum DEX Landscape"*.

## What's Inside

This notebook builds up the DEX routing optimization problem in four stages:

1. **AMM Primitives** — constant-product exchange functions, state updates, price impact
2. **Graph Routing** — shortest-path (Bellman-Ford) on the token graph
3. **Split Routing** — concave optimization of split fractions across parallel pools
4. **Gas-Aware Routing** — mixed-integer formulation with fixed costs per pool

All sections are interactive with sliders — adjust parameters and see results update in real time.

## Running Locally

Install [Julia](https://julialang.org/downloads/) (≥ 1.10), then:

```julia
using Pkg
Pkg.add("Pluto")
import Pluto
Pluto.run()
```

Open `dex_routing.jl` from the Pluto file picker. Pluto will automatically install all dependencies on first run.

## Repository Structure

```
├── binder/
│   ├── Project.toml   # Pluto dependency for Binder
│   └── start          # Launch script for Binder
├── dex_routing.jl     # The interactive Pluto notebook
└── README.md          # This file
```
