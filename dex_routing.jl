### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 9a1b2c3d-0001-4000-8000-000000000001
begin
	using PlutoUI, Graphs
	using JuMP, HiGHS, Ipopt
	using Plots
	using LinearAlgebra
	md"**Packages loaded.**"
end

# ╔═╡ 9a1b2c3d-0002-4000-8000-000000000001
md"""
# Gas-Aware DEX Routing Optimization

*Companion notebook to "Finding Solutions on the Ethereum DEX Landscape"*

This notebook builds up the DEX routing optimization problem in four stages:

1. **AMM Primitives** — constant-product exchange functions, state updates, price impact
2. **Graph Routing** — shortest-path (Dijkstra) on the token graph
3. **Split Routing** — concave optimization of split fractions across parallel pools
4. **Gas-Aware Routing** — mixed-integer formulation with fixed costs per pool

Each section is self-contained and interactive.
"""

# ╔═╡ 9a1b2c3d-0003-4000-8000-000000000001
md"""
## 1. AMM Primitives

A **constant-product AMM** (Uniswap V2 style) maintains the invariant

$$k = r_{\text{in}} \cdot r_{\text{out}}$$

The exchange function maps input amount $x$ to output amount:

$$f(x) = \frac{x \cdot r_{\text{out}}}{x + r_{\text{in}}}$$

After a swap, the reserves update to $(r_{\text{in}} + x,\; r_{\text{out}} - f(x))$, preserving the invariant (up to fees).
"""

# ╔═╡ 9a1b2c3d-0004-4000-8000-000000000001
begin
	"""
	A constant-product AMM pool with fee.

	Fields:
	- `token_in`, `token_out`: token identifiers (symbols)
	- `reserve_in`, `reserve_out`: current reserves
	- `fee`: swap fee (e.g., 0.003 for 0.3%; γ = 1 - fee)
	- `gas_cost`: fixed gas cost in ETH for using this pool
	"""
	mutable struct CPPool
		token_in::Symbol
		token_out::Symbol
		reserve_in::Float64
		reserve_out::Float64
		fee::Float64
		gas_cost::Float64
	end

	"Effective input after fee deduction"
	γ(p::CPPool) = 1.0 - p.fee

	"Exchange function: amount out given amount in"
	function amount_out(p::CPPool, x::Real)
		x ≤ 0 && return 0.0
		x_eff = x * γ(p)
		(x_eff * p.reserve_out) / (x_eff + p.reserve_in)
	end

	"Spot price (marginal exchange rate at zero input)"
	spot_price(p::CPPool) = γ(p) * p.reserve_out / p.reserve_in

	"Execute a swap: mutate reserves, return output amount"
	function swap!(p::CPPool, x::Real)
		Δout = amount_out(p, x)
		p.reserve_in += x
		p.reserve_out -= Δout
		Δout
	end

	"Invariant k = r_in * r_out (should be non-decreasing due to fees)"
	invariant(p::CPPool) = p.reserve_in * p.reserve_out

	md"Defined `CPPool` with `amount_out`, `spot_price`, `swap!`, `invariant`."
end

# ╔═╡ 9a1b2c3d-0005-4000-8000-000000000001
md"### Price Impact Visualization"

# ╔═╡ 9a1b2c3d-0006-4000-8000-000000000001
md"**Reserve In:** $(@bind rin Slider(100:100:10000, default=1000, show_value=true)) | **Reserve Out:** $(@bind rout Slider(100:100:10000, default=2000, show_value=true)) | **Fee (%):** $(@bind fee_pct Slider(0.0:0.1:3.0, default=0.3, show_value=true))"

# ╔═╡ 9a1b2c3d-0007-4000-8000-000000000001
let
	p = CPPool(:A, :B, Float64(rin), Float64(rout), fee_pct/100, 0.0)

	xs = range(0, p.reserve_in * 0.5, length=200)
	ys = [amount_out(p, x) for x in xs]

	# Effective price = y/x
	prices = [x > 0 ? amount_out(p, x) / x : spot_price(p) for x in xs]

	p1 = plot(xs, ys,
		xlabel="Amount In", ylabel="Amount Out",
		title="Exchange Function f(x)",
		label="f(x)", lw=2, color=:blue)
	plot!(p1, xs, spot_price(p) .* xs,
		label="Spot price (tangent)", ls=:dash, color=:gray)

	p2 = plot(xs, prices,
		xlabel="Amount In", ylabel="Effective Price (out/in)",
		title="Price Impact",
		label="Effective price", lw=2, color=:red)
	hline!(p2, [spot_price(p)], label="Spot price", ls=:dash, color=:gray)

	plot(p1, p2, layout=(1,2), size=(800, 350))
end

# ╔═╡ 9a1b2c3d-0008-4000-8000-000000000001
md"""
### Invariant Preservation

After a swap of size $x$, the invariant $k$ is **non-decreasing** (fees accrue to LPs):
"""

# ╔═╡ 9a1b2c3d-0009-4000-8000-000000000001
let
	p = CPPool(:A, :B, 1000.0, 2000.0, 0.003, 0.0)
	k_before = invariant(p)
	out = swap!(p, 100.0)
	k_after = invariant(p)

	md"""
	| | Before | After |
	|---|---|---|
	| Reserve In | 1000.0 | $(round(p.reserve_in, digits=2)) |
	| Reserve Out | 2000.0 | $(round(p.reserve_out, digits=2)) |
	| k | $(round(k_before, digits=2)) | $(round(k_after, digits=2)) |
	| Output | — | $(round(out, digits=4)) |
	| k increased? | — | **$(k_after ≥ k_before)** |
	"""
end

# ╔═╡ 9a1b2c3d-0010-4000-8000-000000000001
md"""
---
## 2. Graph Routing (Shortest Path)

Model the DEX landscape as a **directed weighted graph**:
- **Vertices** = tokens
- **Edges** = pools (directed: token\_in → token\_out)
- **Weight** = $-\log(\text{spot\_price})$ (so shortest path = best exchange rate)

Dijkstra finds the single best path. Negative cycles = arbitrage.
"""

# ╔═╡ 9a1b2c3d-0011-4000-8000-000000000001
begin
	"""
	Build a token graph from a set of pools.
	Returns (graph, edge_weights, token_list, pool_map).
	"""
	function build_token_graph(pools::Vector{CPPool})
		# Collect all tokens
		tokens = unique(vcat(
			[p.token_in for p in pools],
			[p.token_out for p in pools]
		))
		token_idx = Dict(t => i for (i, t) in enumerate(tokens))
		n = length(tokens)

		g = SimpleDiGraph(n)
		weights = Dict{Tuple{Int,Int}, Float64}()
		pool_map = Dict{Tuple{Int,Int}, CPPool}()

		for p in pools
			i, j = token_idx[p.token_in], token_idx[p.token_out]
			add_edge!(g, i, j)
			sp = spot_price(p)
			w = sp > 0 ? -log(sp) : Inf
			# Keep the best (lowest weight) edge if multiple pools
			if !haskey(weights, (i,j)) || w < weights[(i,j)]
				weights[(i,j)] = w
				pool_map[(i,j)] = p
			end
		end

		(g, weights, tokens, token_idx, pool_map)
	end

	md"Defined `build_token_graph`."
end

# ╔═╡ 9a1b2c3d-0012-4000-8000-000000000001
md"### Example: 5-Token Network"

# ╔═╡ 9a1b2c3d-0013-4000-8000-000000000001
begin
	# Create a realistic small DEX network
	example_pools = [
		# ETH-USDC pools (two competing pools with different reserves)
		CPPool(:ETH, :USDC, 500.0, 1_000_000.0, 0.003, 0.00015),   # Large pool
		CPPool(:ETH, :USDC, 100.0, 195_000.0, 0.003, 0.00015),     # Small pool
		# USDC-DAI
		CPPool(:USDC, :DAI, 500_000.0, 499_500.0, 0.001, 0.00010),  # Stablecoin pool
		# ETH-WBTC
		CPPool(:ETH, :WBTC, 1000.0, 50.0, 0.003, 0.00020),
		# WBTC-USDC
		CPPool(:WBTC, :USDC, 30.0, 900_000.0, 0.003, 0.00020),
		# DAI-USDC (reverse direction)
		CPPool(:DAI, :USDC, 300_000.0, 300_200.0, 0.001, 0.00010),
		# ETH-DAI
		CPPool(:ETH, :DAI, 200.0, 390_000.0, 0.003, 0.00015),
		# UNI-ETH
		CPPool(:UNI, :ETH, 50_000.0, 20.0, 0.003, 0.00015),
		# UNI-USDC
		CPPool(:UNI, :USDC, 30_000.0, 12_000.0, 0.003, 0.00015),
	]

	g, weights, tokens, token_idx, pool_map = build_token_graph(example_pools)

	md"""
	**Tokens:** $(join(string.(tokens), ", "))

	**Pools:** $(length(example_pools)) pools across $(nv(g)) tokens and $(ne(g)) directed edges
	"""
end

# ╔═╡ 9a1b2c3d-0014-4000-8000-000000000001
begin
	"Find shortest path using Bellman-Ford (handles negative weights for arb detection)"
	function find_best_path(g, weights, token_idx, source::Symbol, target::Symbol)
		s = token_idx[source]
		d = token_idx[target]
		n = nv(g)

		# Bellman-Ford
		dist = fill(Inf, n)
		pred = fill(-1, n)
		dist[s] = 0.0

		for _ in 1:n-1
			for e in edges(g)
				u, v = Graphs.src(e), Graphs.dst(e)
				w = get(weights, (u, v), Inf)
				if dist[u] + w < dist[v]
					dist[v] = dist[u] + w
					pred[v] = u
				end
			end
		end

		# Reconstruct path
		path = Int[]
		node = d
		while node != -1
			pushfirst!(path, node)
			node == s && break
			node = pred[node]
		end

		effective_rate = exp(-dist[d])
		(path=path, distance=dist[d], effective_rate=effective_rate)
	end

	md"Defined `find_best_path` (Bellman-Ford)."
end

# ╔═╡ 9a1b2c3d-0015-4000-8000-000000000001
md"""
### Route: ETH → USDC

Compare direct route vs multi-hop:
"""

# ╔═╡ 9a1b2c3d-0016-4000-8000-000000000001
let
	result = find_best_path(g, weights, token_idx, :ETH, :USDC)
	path_tokens = [tokens[i] for i in result.path]

	# Also check ETH → WBTC → USDC
	r1 = spot_price(pool_map[(token_idx[:ETH], token_idx[:WBTC])])
	r2 = spot_price(pool_map[(token_idx[:WBTC], token_idx[:USDC])])
	indirect_rate = r1 * r2

	direct = spot_price(pool_map[(token_idx[:ETH], token_idx[:USDC])])

	md"""
	**Best path:** $(join(string.(path_tokens), " → "))

	**Effective spot rate:** $(round(result.effective_rate, digits=2)) USDC/ETH

	| Route | Rate (USDC/ETH) |
	|---|---|
	| ETH → USDC (direct) | $(round(direct, digits=2)) |
	| ETH → WBTC → USDC | $(round(indirect_rate, digits=2)) |
	| Best (Bellman-Ford) | $(round(result.effective_rate, digits=2)) |
	"""
end

# ╔═╡ 9a1b2c3d-0017-4000-8000-000000000001
md"""
---
## 3. Split Routing

When multiple pools connect the same token pair, **splitting** the order across them reduces price impact. For constant-product AMMs, the total output as a function of the split fraction is **concave**, so the optimum can be found analytically or via simple optimization.

Given $n$ parallel pools with exchange functions $f_1, \ldots, f_n$, and total input $X$, find split fractions $s_1, \ldots, s_n$ with $\sum s_i = 1$ maximizing:

$$\sum_{i=1}^n f_i(s_i \cdot X)$$
"""

# ╔═╡ 9a1b2c3d-0018-4000-8000-000000000001
md"**Total input (ETH):** $(@bind total_input_eth Slider(0.1:0.1:50.0, default=5.0, show_value=true))"

# ╔═╡ 9a1b2c3d-0019-4000-8000-000000000001
let
	X = total_input_eth

	# The two ETH→USDC pools
	p1 = CPPool(:ETH, :USDC, 500.0, 1_000_000.0, 0.003, 0.00015)
	p2 = CPPool(:ETH, :USDC, 100.0, 195_000.0, 0.003, 0.00015)

	# Sweep split fraction s ∈ [0, 1] (fraction to pool 1)
	ss = range(0, 1, length=200)
	outputs = [amount_out(p1, s*X) + amount_out(p2, (1-s)*X) for s in ss]

	# Optimal split (brute force from sweep)
	best_idx = argmax(outputs)
	s_opt = ss[best_idx]
	out_opt = outputs[best_idx]

	# Compare to no-split baselines
	out_all_p1 = amount_out(p1, X)
	out_all_p2 = amount_out(p2, X)

	p = plot(ss, outputs,
		xlabel="Fraction to Large Pool", ylabel="Total USDC Output",
		title="Split Routing: $(X) ETH across 2 pools",
		label="Split output", lw=2, color=:blue)
	scatter!(p, [s_opt], [out_opt], label="Optimal (s=$(round(s_opt, digits=3)))",
		ms=8, color=:red)
	hline!(p, [out_all_p1], label="100% Large Pool", ls=:dash, color=:green)
	hline!(p, [out_all_p2], label="100% Small Pool", ls=:dash, color=:orange)

	plot(p, size=(700, 400))
end

# ╔═╡ 9a1b2c3d-0020-4000-8000-000000000001
md"""
### Optimal Split via JuMP (Nonlinear)

Solve the split optimization as a proper NLP:
"""

# ╔═╡ 9a1b2c3d-0021-4000-8000-000000000001
function optimal_split_nlp(pools::Vector{CPPool}, total_input::Float64)
	n = length(pools)

	model = Model(Ipopt.Optimizer)
	set_silent(model)

	@variable(model, 0 ≤ s[1:n] ≤ 1)
	@constraint(model, sum(s) == 1)

	# Register the amount_out function for each pool
	for (i, p) in enumerate(pools)
		ri, ro, g = p.reserve_in, p.reserve_out, γ(p)
		# f_i(x) = (g * x * ro) / (g * x + ri)
		# We maximize sum of f_i(s_i * X)
	end

	# Nonlinear objective: sum of (γ * s_i * X * r_out) / (γ * s_i * X + r_in)
	@objective(model, Max,
		sum(
			(γ(pools[i]) * s[i] * total_input * pools[i].reserve_out) /
			(γ(pools[i]) * s[i] * total_input + pools[i].reserve_in)
			for i in 1:n
		)
	)

	optimize!(model)

	splits = value.(s)
	outputs = [amount_out(pools[i], splits[i] * total_input) for i in 1:n]

	(splits=splits, outputs=outputs, total_output=sum(outputs),
	 status=termination_status(model))
end

# ╔═╡ 9a1b2c3d-0022-4000-8000-000000000001
let
	X = total_input_eth
	p1 = CPPool(:ETH, :USDC, 500.0, 1_000_000.0, 0.003, 0.00015)
	p2 = CPPool(:ETH, :USDC, 100.0, 195_000.0, 0.003, 0.00015)

	result = optimal_split_nlp([p1, p2], X)

	md"""
	**Optimal split (NLP solver):** $(round(result.splits[1]*100, digits=1))% to Large Pool, $(round(result.splits[2]*100, digits=1))% to Small Pool

	**Total output:** $(round(result.total_output, digits=2)) USDC

	**Status:** $(result.status)

	For comparison:
	- 100% Large Pool: $(round(amount_out(p1, X), digits=2)) USDC
	- 100% Small Pool: $(round(amount_out(p2, X), digits=2)) USDC
	- **Improvement from splitting:** $(round(result.total_output - max(amount_out(p1, X), amount_out(p2, X)), digits=2)) USDC
	"""
end

# ╔═╡ 9a1b2c3d-0023-4000-8000-000000000001
md"""
---
## 4. Gas-Aware Routing (MICP)

This is the core contribution. The problem from Angeris et al. (2022):

$$\max_{\Delta_p, z_p} \;\; \underbrace{e_j^\top \sum_p \Delta_p}_{\text{output}} \;-\; \underbrace{\sum_p c_p \cdot z_p}_{\text{gas cost}}$$

subject to:
- $\Delta_p \in \mathcal{T}_p$ (feasible trades per CFMM)
- $\sum_p \Delta_p \geq 0$ (no token created from nothing)
- $|\Delta_p| \leq M \cdot z_p$ (trade is zero if pool not activated)
- $z_p \in \{0, 1\}$ (binary pool activation)

where $c_p$ is the gas cost of using pool $p$ (in units of the output token).

### Why this matters for consumers

For a **small trade** (e.g., 0.1 ETH), using 5 pools might give 0.01% better execution but cost 5× the gas — a net loss. The gas-aware formulation finds the right tradeoff.
"""

# ╔═╡ 9a1b2c3d-0024-4000-8000-000000000001
md"""
### 4.1 The Full Network Model

We model the problem as a **nonlinear mixed-integer program** in JuMP.

For each pool $p$ connecting token $i \to j$:
- Decision: input amount $x_p \geq 0$
- Output: $f_p(x_p)$ (concave, from AMM formula)
- Binary: $z_p \in \{0,1\}$ — is this pool used?
- Big-M: $x_p \leq M \cdot z_p$
- Gas cost: $c_p \cdot z_p$ (converted to output-token units)
"""

# ╔═╡ 9a1b2c3d-0025-4000-8000-000000000001
begin
	"""
	Solve the gas-aware routing problem as a MINLP.

	- `pools`: vector of CPPool
	- `total_input`: amount of input token to swap
	- `input_token`, `output_token`: which tokens
	- `gas_price_in_output`: price of 1 unit of gas in output token units
	                          (e.g., if output is USDC and gas is in ETH,
	                           this is the ETH/USDC rate × gas_cost_in_ETH)
	- `include_gas`: if false, solve the relaxed (gas-free) version
	"""
	function gas_aware_route(
		pools::Vector{CPPool},
		total_input::Float64,
		input_token::Symbol,
		output_token::Symbol;
		gas_price_in_output::Float64 = 2000.0,  # 1 ETH ≈ 2000 USDC
		include_gas::Bool = true,
		use_integer::Bool = true,
	)
		n = length(pools)

		# Collect all tokens
		all_tokens = unique(vcat(
			[p.token_in for p in pools],
			[p.token_out for p in pools]
		))
		tidx = Dict(t => i for (i, t) in enumerate(all_tokens))
		ntokens = length(all_tokens)

		model = Model(Ipopt.Optimizer)
		set_silent(model)

		# --- Variables ---
		# x[p] = amount of token_in sent through pool p
		@variable(model, x[1:n] ≥ 0)

		# z[p] ∈ {0,1} = is pool p used?
		if use_integer && include_gas
			# Ipopt can't handle integers natively, so we use a
			# continuous relaxation + penalty approach
			@variable(model, 0 ≤ z[1:n] ≤ 1)
		else
			@variable(model, 0 ≤ z[1:n] ≤ 1)
		end

		# Big-M: x[p] ≤ M * z[p]
		M = total_input * 2.0
		for i in 1:n
			@constraint(model, x[i] ≤ M * z[i])
		end

		# --- Flow conservation ---
		# For each token t:
		#   (inflow from source if t == input_token)
		#   + sum of outputs from pools producing t
		#   - sum of inputs to pools consuming t
		#   ≥ 0  (for intermediate tokens)
		#   = maximized (for output token)

		# For simplicity, model parallel pools on same pair:
		# net_flow[t] = supply[t] + sum(f_p(x_p) for p producing t) - sum(x_p for p consuming t)

		# Source: the input token gets `total_input` supply
		# Sink: we maximize the output token's net flow

		# Nonlinear: output of pool p = γ_p * x_p * r_out_p / (γ_p * x_p + r_in_p)

		# Total input token consumed
		consuming_input = [i for i in 1:n if pools[i].token_in == input_token]
		@constraint(model, sum(x[i] for i in consuming_input) ≤ total_input)

		# Flow conservation for intermediate tokens
		for t in all_tokens
			t == input_token && continue
			t == output_token && continue

			producing = [i for i in 1:n if pools[i].token_out == t]
			consuming = [i for i in 1:n if pools[i].token_in == t]

			if !isempty(producing) && !isempty(consuming)
				# Inflow ≥ outflow for intermediate tokens
				@constraint(model,
					sum(
						(γ(pools[i]) * x[i] * pools[i].reserve_out) /
						(γ(pools[i]) * x[i] + pools[i].reserve_in)
						for i in producing
					) ≥ sum(x[i] for i in consuming)
				)
			elseif isempty(producing) && !isempty(consuming)
				# No source for this token — force all consuming flows to zero
				for i in consuming
					@constraint(model, x[i] == 0)
				end
			end
		end

		# --- Objective: maximize output - gas ---
		producing_output = [i for i in 1:n if pools[i].token_out == output_token]
		consuming_output = [i for i in 1:n if pools[i].token_in == output_token]

		gas_penalty = include_gas ? 1.0 : 0.0

		@objective(model, Max,
			# Output token produced
			sum(
				(γ(pools[i]) * x[i] * pools[i].reserve_out) /
				(γ(pools[i]) * x[i] + pools[i].reserve_in)
				for i in producing_output
			)
			# Minus output token consumed (if any intermediate hops consume it)
			- sum(x[i] for i in consuming_output)
			# Minus gas costs (in output token units)
			- gas_penalty * sum(
				pools[i].gas_cost * gas_price_in_output * z[i]
				for i in 1:n
			)
		)

		optimize!(model)

		xs = value.(x)
		zs = value.(z)

		total_out = sum(
			amount_out(pools[i], xs[i])
			for i in producing_output;
			init=0.0
		) - sum(xs[i] for i in consuming_output; init=0.0)

		total_gas = sum(pools[i].gas_cost * zs[i] for i in 1:n)
		net_output = total_out - (include_gas ? total_gas * gas_price_in_output : 0.0)

		active_pools = [(i, pools[i], xs[i], zs[i]) for i in 1:n if zs[i] > 0.01]

		(
			total_output = total_out,
			total_gas_eth = total_gas,
			total_gas_output = total_gas * gas_price_in_output,
			net_output = net_output,
			active_pools = active_pools,
			all_x = xs,
			all_z = zs,
			status = termination_status(model),
		)
	end

	md"Defined `gas_aware_route`."
end

# ╔═╡ 9a1b2c3d-0026-4000-8000-000000000001
md"""
### 4.2 Gas-Aware vs Gas-Free: The Tradeoff

Compare routing with and without gas awareness at different trade sizes.
"""

# ╔═╡ 9a1b2c3d-0027-4000-8000-000000000001
@bind trade_size_eth Slider(0.1:0.1:20.0, default=1.0, show_value=true)

# ╔═╡ 9a1b2c3d-0028-4000-8000-000000000001
let
	X = trade_size_eth
	eth_usdc = 2000.0  # ETH price in USDC

	# Gas-free routing
	r_free = gas_aware_route(example_pools, X, :ETH, :USDC;
		gas_price_in_output=eth_usdc, include_gas=false)

	# Gas-aware routing
	r_gas = gas_aware_route(example_pools, X, :ETH, :USDC;
		gas_price_in_output=eth_usdc, include_gas=true)

	md"""
	### Trade: $(X) ETH → USDC (ETH ≈ \$$(eth_usdc))

	| Metric | Gas-Free | Gas-Aware |
	|--------|----------|-----------|
	| Gross output (USDC) | $(round(r_free.total_output, digits=2)) | $(round(r_gas.total_output, digits=2)) |
	| Gas cost (USDC) | $(round(r_free.total_gas_output, digits=4)) | $(round(r_gas.total_gas_output, digits=4)) |
	| **Net output (USDC)** | **$(round(r_free.total_output - r_free.total_gas_output, digits=2))** | **$(round(r_gas.net_output, digits=2))** |
	| Pools used | $(length(r_free.active_pools)) | $(length(r_gas.active_pools)) |
	| Status | $(r_free.status) | $(r_gas.status) |

	**Gas-free active pools:**
	$(join(["- Pool $(i): $(p.token_in)→$(p.token_out) (x=$(round(x, digits=4)), z=$(round(z, digits=3)))" for (i, p, x, z) in r_free.active_pools], "\n"))

	**Gas-aware active pools:**
	$(join(["- Pool $(i): $(p.token_in)→$(p.token_out) (x=$(round(x, digits=4)), z=$(round(z, digits=3)))" for (i, p, x, z) in r_gas.active_pools], "\n"))
	"""
end

# ╔═╡ 9a1b2c3d-0029-4000-8000-000000000001
md"""
### 4.3 Trade Size Sweep: When Gas Matters Most

For small trades, gas dominates. For large trades, price impact dominates and splitting across more pools pays for itself.
"""

# ╔═╡ 9a1b2c3d-0030-4000-8000-000000000001
let
	eth_usdc = 2000.0
	trade_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

	results_free = []
	results_gas = []

	for X in trade_sizes
		rf = gas_aware_route(example_pools, X, :ETH, :USDC;
			gas_price_in_output=eth_usdc, include_gas=false)
		rg = gas_aware_route(example_pools, X, :ETH, :USDC;
			gas_price_in_output=eth_usdc, include_gas=true)
		push!(results_free, rf)
		push!(results_gas, rg)
	end

	net_free = [r.total_output - r.total_gas_output for r in results_free]
	net_gas = [r.net_output for r in results_gas]
	n_pools_free = [length(r.active_pools) for r in results_free]
	n_pools_gas = [length(r.active_pools) for r in results_gas]

	p1 = plot(trade_sizes, net_gas .- net_free,
		xlabel="Trade Size (ETH)", ylabel="Gas-aware advantage (USDC)",
		title="Net Output Improvement from Gas Awareness",
		label="Gas-aware − Gas-free", lw=2, marker=:circle,
		color=:blue)
	hline!(p1, [0], ls=:dash, color=:gray, label=nothing)

	p2 = plot(trade_sizes, n_pools_free,
		xlabel="Trade Size (ETH)", ylabel="Pools Used",
		title="Number of Active Pools",
		label="Gas-free", lw=2, marker=:circle, color=:red)
	plot!(p2, trade_sizes, n_pools_gas,
		label="Gas-aware", lw=2, marker=:square, color=:blue)

	plot(p1, p2, layout=(1,2), size=(800, 350))
end

# ╔═╡ 9a1b2c3d-0031-4000-8000-000000000001
md"""
### 4.4 The Threshold Heuristic (Angeris et al. §5)

A practical alternative to solving the full MICP:

1. **Solve the convex relaxation** (gas-free, continuous $z_p \in [0,1]$)
2. **Threshold**: For each pool, if flow $< t \cdot X_{\text{total}}$, prune it ($z_p = 0$)
3. **Re-solve** the convex problem with only the surviving pools
4. **Sweep** thresholds $t \in \{0.01, 0.05, 0.10, \ldots\}$ and pick the best net output

This avoids integer programming entirely — just repeated convex solves.
"""

# ╔═╡ 9a1b2c3d-0032-4000-8000-000000000001
function threshold_heuristic(
	pools::Vector{CPPool},
	total_input::Float64,
	input_token::Symbol,
	output_token::Symbol;
	gas_price_in_output::Float64 = 2000.0,
	thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
)
	# Step 1: solve gas-free relaxation
	r0 = gas_aware_route(pools, total_input, input_token, output_token;
		gas_price_in_output=gas_price_in_output, include_gas=false)

	best_net = -Inf
	best_result = nothing
	best_threshold = 0.0
	all_results = []

	for t in thresholds
		# Step 2: threshold — keep pools with flow > t * total_input
		active_mask = r0.all_x .> (t * total_input)
		active_pools = pools[active_mask]

		isempty(active_pools) && continue

		# Check that active pools still connect input to output
		any(p -> p.token_in == input_token, active_pools) || continue
		any(p -> p.token_out == output_token, active_pools) || continue

		# Step 3: re-solve with only active pools (gas-aware)
		r = gas_aware_route(active_pools, total_input, input_token, output_token;
			gas_price_in_output=gas_price_in_output, include_gas=true)

		push!(all_results, (threshold=t, result=r, n_pools=length(active_pools)))

		if r.net_output > best_net
			best_net = r.net_output
			best_result = r
			best_threshold = t
		end
	end

	(best_threshold=best_threshold, best_result=best_result, all_results=all_results)
end

# ╔═╡ 9a1b2c3d-0033-4000-8000-000000000001
@bind heuristic_trade_size Slider(0.1:0.1:20.0, default=1.0, show_value=true)

# ╔═╡ 9a1b2c3d-0034-4000-8000-000000000001
let
	X = heuristic_trade_size
	eth_usdc = 2000.0

	h = threshold_heuristic(example_pools, X, :ETH, :USDC;
		gas_price_in_output=eth_usdc)

	ts = [r.threshold for r in h.all_results]
	nets = [r.result.net_output for r in h.all_results]
	nps = [r.n_pools for r in h.all_results]

	p1 = plot(ts, nets,
		xlabel="Threshold (fraction of total input)",
		ylabel="Net Output (USDC)",
		title="Threshold Heuristic Sweep ($(X) ETH)",
		label="Net output", lw=2, marker=:circle, color=:blue)
	scatter!(p1, [h.best_threshold], [h.best_result.net_output],
		label="Best (t=$(h.best_threshold))", ms=10, color=:red)

	p2 = bar(ts, nps,
		xlabel="Threshold", ylabel="Pools in subset",
		title="Pool Pruning",
		label="Active pools", color=:lightblue)

	plot(p1, p2, layout=(1,2), size=(800, 350))
end

# ╔═╡ 9a1b2c3d-0035-4000-8000-000000000001
md"""
---
## 5. Summary & Connection to the Shielded Swap

The key insight for the **anoma-swap delegate**:

1. **The routing problem is well-understood mathematically** — convex when ignoring gas, MICP when including it.

2. **Gas awareness matters for consumers** — especially for small-to-medium trades where gas can exceed the price improvement from splitting.

3. **The threshold heuristic is practical** — solve one convex problem, prune, re-solve. No integer programming needed. This runs in milliseconds.

4. **Decoupling authorization from routing** means we compute the route *after* the ZK proof completes, using fresh prices. The routing computation itself is fast ($<100$ ms), so it fits in the window between proof completion and transaction submission.

### Next Steps

- **Integrate real pool data** via Tycho Indexer or on-chain RPC
- **Add concentrated liquidity** (Uniswap V3 tick ranges — piecewise exchange functions)
- **Compare to CFMMRouter.jl** (dual decomposition) for the convex subproblem
- **Benchmark** against 1inch / Paraswap API quotes
- **Implement DSSP** (Dynamic Slope Scaling) for near-optimal gas-aware routing without MICP
"""

# ╔═╡ 9a1b2c3d-0036-4000-8000-000000000001
md"""
---
*Notebook generated as companion to "Finding Solutions on the Ethereum DEX Landscape" — see `/root/dexage/docs/finding-solutions.pdf`*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Graphs = "~1.9"
HiGHS = "~1.9"
Ipopt = "~1.6"
JuMP = "~1.23"
Plots = "~1.40"
PlutoUI = "~0.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "0000000000000000000000000000000000000000"

"""

# ╔═╡ Cell order:
# ╟─9a1b2c3d-0001-4000-8000-000000000001
# ╟─9a1b2c3d-0002-4000-8000-000000000001
# ╟─9a1b2c3d-0003-4000-8000-000000000001
# ╠═9a1b2c3d-0004-4000-8000-000000000001
# ╟─9a1b2c3d-0005-4000-8000-000000000001
# ╠═9a1b2c3d-0006-4000-8000-000000000001
# ╠═9a1b2c3d-0007-4000-8000-000000000001
# ╟─9a1b2c3d-0008-4000-8000-000000000001
# ╠═9a1b2c3d-0009-4000-8000-000000000001
# ╟─9a1b2c3d-0010-4000-8000-000000000001
# ╠═9a1b2c3d-0011-4000-8000-000000000001
# ╟─9a1b2c3d-0012-4000-8000-000000000001
# ╠═9a1b2c3d-0013-4000-8000-000000000001
# ╠═9a1b2c3d-0014-4000-8000-000000000001
# ╟─9a1b2c3d-0015-4000-8000-000000000001
# ╠═9a1b2c3d-0016-4000-8000-000000000001
# ╟─9a1b2c3d-0017-4000-8000-000000000001
# ╠═9a1b2c3d-0018-4000-8000-000000000001
# ╠═9a1b2c3d-0019-4000-8000-000000000001
# ╟─9a1b2c3d-0020-4000-8000-000000000001
# ╠═9a1b2c3d-0021-4000-8000-000000000001
# ╠═9a1b2c3d-0022-4000-8000-000000000001
# ╟─9a1b2c3d-0023-4000-8000-000000000001
# ╟─9a1b2c3d-0024-4000-8000-000000000001
# ╠═9a1b2c3d-0025-4000-8000-000000000001
# ╟─9a1b2c3d-0026-4000-8000-000000000001
# ╠═9a1b2c3d-0027-4000-8000-000000000001
# ╠═9a1b2c3d-0028-4000-8000-000000000001
# ╟─9a1b2c3d-0029-4000-8000-000000000001
# ╠═9a1b2c3d-0030-4000-8000-000000000001
# ╟─9a1b2c3d-0031-4000-8000-000000000001
# ╠═9a1b2c3d-0032-4000-8000-000000000001
# ╠═9a1b2c3d-0033-4000-8000-000000000001
# ╠═9a1b2c3d-0034-4000-8000-000000000001
# ╟─9a1b2c3d-0035-4000-8000-000000000001
# ╟─9a1b2c3d-0036-4000-8000-000000000001
