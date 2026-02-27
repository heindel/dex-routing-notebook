#!/usr/bin/env julia
# End-to-end test of the DEX routing notebook logic
# Runs all sections non-interactively

println("=" ^ 60)
println("DEX Routing Notebook — End-to-End Tests")
println("=" ^ 60)

# ── Packages ──────────────────────────────────────────────
println("\n[1/5] Loading packages...")
import Pkg
Pkg.activate(temp=true)
Pkg.add(["Graphs", "JuMP", "Ipopt"]; io=devnull)
using Graphs, JuMP, Ipopt, LinearAlgebra

println("  ✓ Packages loaded")

# ══════════════════════════════════════════════════════════
# Section 1: AMM Primitives
# ══════════════════════════════════════════════════════════
println("\n[2/5] Section 1: AMM Primitives")

mutable struct CPPool
    token_in::Symbol
    token_out::Symbol
    reserve_in::Float64
    reserve_out::Float64
    fee::Float64
    gas_cost::Float64
end

γ(p::CPPool) = 1.0 - p.fee

function amount_out(p::CPPool, x::Real)
    x ≤ 0 && return 0.0
    x_eff = x * γ(p)
    (x_eff * p.reserve_out) / (x_eff + p.reserve_in)
end

spot_price(p::CPPool) = γ(p) * p.reserve_out / p.reserve_in

function swap!(p::CPPool, x::Real)
    Δout = amount_out(p, x)
    p.reserve_in += x
    p.reserve_out -= Δout
    Δout
end

invariant(p::CPPool) = p.reserve_in * p.reserve_out

# Test 1.1: basic exchange
p = CPPool(:ETH, :USDC, 1000.0, 2_000_000.0, 0.003, 0.0)
out = amount_out(p, 1.0)
sp = spot_price(p)
@assert out > 0 "amount_out should be positive"
@assert sp ≈ 0.997 * 2000.0 atol=0.1 "spot price ≈ γ × 2000"
println("  ✓ amount_out(1 ETH) = $(round(out, digits=2)) USDC, spot = $(round(sp, digits=2))")

# Test 1.2: concavity — f(2x) < 2·f(x)
out1 = amount_out(p, 10.0)
out2 = amount_out(p, 20.0)
@assert out2 < 2 * out1 "exchange function must be concave"
println("  ✓ Concavity: f(20) = $(round(out2, digits=2)) < 2·f(10) = $(round(2*out1, digits=2))")

# Test 1.3: invariant non-decreasing
p2 = CPPool(:ETH, :USDC, 1000.0, 2_000_000.0, 0.003, 0.0)
k_before = invariant(p2)
swap!(p2, 50.0)
k_after = invariant(p2)
@assert k_after ≥ k_before "invariant must be non-decreasing (fees accrue)"
println("  ✓ Invariant: $(round(k_before, digits=0)) → $(round(k_after, digits=0)) (Δ = +$(round(k_after - k_before, digits=0)))")

# Test 1.4: zero and negative input
@assert amount_out(p, 0.0) == 0.0 "zero input → zero output"
@assert amount_out(p, -5.0) == 0.0 "negative input → zero output"
println("  ✓ Edge cases: f(0) = 0, f(-5) = 0")

# ══════════════════════════════════════════════════════════
# Section 2: Graph Routing
# ══════════════════════════════════════════════════════════
println("\n[3/5] Section 2: Graph Routing")

function build_token_graph(pools::Vector{CPPool})
    tokens = unique(vcat([p.token_in for p in pools], [p.token_out for p in pools]))
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
        if !haskey(weights, (i,j)) || w < weights[(i,j)]
            weights[(i,j)] = w
            pool_map[(i,j)] = p
        end
    end
    (g, weights, tokens, token_idx, pool_map)
end

function find_best_path(g, weights, token_idx, source::Symbol, target::Symbol)
    s = token_idx[source]
    d = token_idx[target]
    n = nv(g)
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

# Build 5-token network
example_pools = [
    CPPool(:ETH, :USDC, 500.0, 1_000_000.0, 0.003, 0.00015),
    CPPool(:ETH, :USDC, 100.0, 195_000.0, 0.003, 0.00015),
    CPPool(:USDC, :DAI, 500_000.0, 499_500.0, 0.001, 0.00010),
    CPPool(:ETH, :WBTC, 1000.0, 50.0, 0.003, 0.00020),
    CPPool(:WBTC, :USDC, 30.0, 900_000.0, 0.003, 0.00020),
    CPPool(:DAI, :USDC, 300_000.0, 300_200.0, 0.001, 0.00010),
    CPPool(:ETH, :DAI, 200.0, 390_000.0, 0.003, 0.00015),
    CPPool(:UNI, :ETH, 50_000.0, 20.0, 0.003, 0.00015),
    CPPool(:UNI, :USDC, 30_000.0, 12_000.0, 0.003, 0.00015),
]

g, weights, tokens, token_idx, pool_map = build_token_graph(example_pools)
println("  ✓ Graph: $(nv(g)) tokens, $(ne(g)) edges")
println("    Tokens: $(join(string.(tokens), ", "))")

# Test route ETH → USDC
result = find_best_path(g, weights, token_idx, :ETH, :USDC)
path_tokens = [tokens[i] for i in result.path]
@assert path_tokens[1] == :ETH "path should start at ETH"
@assert path_tokens[end] == :USDC "path should end at USDC"
@assert result.effective_rate > 0 "rate should be positive"
println("  ✓ ETH→USDC: $(join(string.(path_tokens), " → ")), rate = $(round(result.effective_rate, digits=2))")

# Test route UNI → DAI (multi-hop)
result2 = find_best_path(g, weights, token_idx, :UNI, :DAI)
path_tokens2 = [tokens[i] for i in result2.path]
@assert length(path_tokens2) ≥ 2 "should find a path"
println("  ✓ UNI→DAI: $(join(string.(path_tokens2), " → ")), rate = $(round(result2.effective_rate, digits=4))")

# ══════════════════════════════════════════════════════════
# Section 3: Split Routing
# ══════════════════════════════════════════════════════════
println("\n[4/5] Section 3: Split Routing (NLP)")

function optimal_split_nlp(pools::Vector{CPPool}, total_input::Float64)
    n = length(pools)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 ≤ s[1:n] ≤ 1)
    @constraint(model, sum(s) == 1)
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
    (splits=splits, outputs=outputs, total_output=sum(outputs), status=termination_status(model))
end

# Two ETH→USDC pools
p_large = CPPool(:ETH, :USDC, 500.0, 1_000_000.0, 0.003, 0.00015)
p_small = CPPool(:ETH, :USDC, 100.0, 195_000.0, 0.003, 0.00015)

for X in [1.0, 5.0, 10.0]
    r = optimal_split_nlp([p_large, p_small], X)
    out_large = amount_out(p_large, X)
    out_small = amount_out(p_small, X)
    best_single = max(out_large, out_small)
    @assert r.total_output ≥ best_single - 0.01 "split should be ≥ best single pool"
    improvement = r.total_output - best_single
    println("  ✓ X=$(X) ETH: split $(round(r.splits[1]*100, digits=1))%/$(round(r.splits[2]*100, digits=1))%, " *
            "output=$(round(r.total_output, digits=2)) USDC, " *
            "improvement=+$(round(improvement, digits=2)) USDC ($(r.status))")
end

# ══════════════════════════════════════════════════════════
# Section 4: Gas-Aware Routing
# ══════════════════════════════════════════════════════════
println("\n[5/5] Section 4: Gas-Aware Routing")

function gas_aware_route(
    pools::Vector{CPPool},
    total_input::Float64,
    input_token::Symbol,
    output_token::Symbol;
    gas_price_in_output::Float64 = 2000.0,
    include_gas::Bool = true,
    use_integer::Bool = true,
)
    n = length(pools)
    all_tokens = unique(vcat([p.token_in for p in pools], [p.token_out for p in pools]))
    tidx = Dict(t => i for (i, t) in enumerate(all_tokens))
    ntokens = length(all_tokens)

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, x[1:n] ≥ 0)
    @variable(model, 0 ≤ z[1:n] ≤ 1)

    M = total_input * 2.0
    for i in 1:n
        @constraint(model, x[i] ≤ M * z[i])
    end

    consuming_input = [i for i in 1:n if pools[i].token_in == input_token]
    @constraint(model, sum(x[i] for i in consuming_input) ≤ total_input)

    for t in all_tokens
        t == input_token && continue
        t == output_token && continue
        producing = [i for i in 1:n if pools[i].token_out == t]
        consuming = [i for i in 1:n if pools[i].token_in == t]
        if !isempty(producing) && !isempty(consuming)
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

    producing_output = [i for i in 1:n if pools[i].token_out == output_token]
    consuming_output = [i for i in 1:n if pools[i].token_in == output_token]

    gas_penalty = include_gas ? 1.0 : 0.0

    @objective(model, Max,
        sum(
            (γ(pools[i]) * x[i] * pools[i].reserve_out) /
            (γ(pools[i]) * x[i] + pools[i].reserve_in)
            for i in producing_output
        )
        - sum(x[i] for i in consuming_output)
        - gas_penalty * sum(pools[i].gas_cost * gas_price_in_output * z[i] for i in 1:n)
    )

    optimize!(model)

    xs = value.(x)
    zs = value.(z)

    total_out = sum(amount_out(pools[i], xs[i]) for i in producing_output; init=0.0) -
                sum(xs[i] for i in consuming_output; init=0.0)
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

eth_usdc_rate = 2000.0

# Test 4.1: gas-free vs gas-aware at small trade
println("\n  4.1 Gas-free vs Gas-aware comparison:")
for X in [0.1, 1.0, 5.0, 10.0]
    rf = gas_aware_route(example_pools, X, :ETH, :USDC;
        gas_price_in_output=eth_usdc_rate, include_gas=false)
    rg = gas_aware_route(example_pools, X, :ETH, :USDC;
        gas_price_in_output=eth_usdc_rate, include_gas=true)

    net_free = rf.total_output - rf.total_gas_output
    advantage = rg.net_output - net_free

    println("    X=$(lpad(string(X), 5)) ETH: " *
            "free=$(length(rf.active_pools)) pools, net=$(round(net_free, digits=2)) | " *
            "gas-aware=$(length(rg.active_pools)) pools, net=$(round(rg.net_output, digits=2)) | " *
            "advantage=$(round(advantage, digits=2)) USDC ($(rg.status))")

    @assert rg.net_output ≥ net_free - 1.0 "gas-aware should be ≥ gas-free (net) within tolerance"
end

# Test 4.2: threshold heuristic
println("\n  4.2 Threshold Heuristic:")

function threshold_heuristic(
    pools::Vector{CPPool},
    total_input::Float64,
    input_token::Symbol,
    output_token::Symbol;
    gas_price_in_output::Float64 = 2000.0,
    thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
)
    r0 = gas_aware_route(pools, total_input, input_token, output_token;
        gas_price_in_output=gas_price_in_output, include_gas=false)

    best_net = -Inf
    best_result = nothing
    best_threshold = 0.0
    all_results = []

    for t in thresholds
        active_mask = r0.all_x .> (t * total_input)
        active_pools = pools[active_mask]
        isempty(active_pools) && continue
        any(p -> p.token_in == input_token, active_pools) || continue
        any(p -> p.token_out == output_token, active_pools) || continue

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

for X in [0.5, 2.0, 10.0]
    h = threshold_heuristic(example_pools, X, :ETH, :USDC;
        gas_price_in_output=eth_usdc_rate)
    @assert h.best_result !== nothing "should find a valid route"
    @assert h.best_result.net_output > 0 "net output should be positive"
    println("    X=$(lpad(string(X), 5)) ETH: best threshold=$(h.best_threshold), " *
            "net=$(round(h.best_result.net_output, digits=2)) USDC, " *
            "pools=$(length(h.best_result.active_pools))")
    for r in h.all_results
        println("      t=$(r.threshold): net=$(round(r.result.net_output, digits=2)), pools=$(r.n_pools)")
    end
end

# ══════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("ALL TESTS PASSED ✓")
println("=" ^ 60)
