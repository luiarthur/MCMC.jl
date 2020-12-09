println("Test metrics")

import Distributions.SpecialFunctions.logabsbeta
import Distributions.StatsBase.harmmean

lbeta(a, b) = logabsbeta(a, b)[1]

@testset "metrics" begin
  @testset "dic" begin
    x = randn(10000)
    ll = normlogpdf.(x)
    @test isfinite(dic(ll))
  end

  @testset "logharmmean" begin
    Random.seed!(0)
    x = randn(100)
    @test log(harmmean(exp.(x))) ≈ MCMC.logharmmeanexp(x) atol=1e-6
  end

  @testset "log_bayes_factor" begin
    Random.seed!(0)
    n = 100
    y = rand(Bernoulli(.7), n)
    sumy = sum(y)

    m0 = Beta(70, 100)
    m1 = Beta(100, 70)

    compute_logc_true(m) = lbeta(m.α + sumy, m.β + n - sumy) - lbeta(m.α, m.β)
    c0_true = compute_logc_true(m0)
    c1_true = compute_logc_true(m1)

    true_log_bf = c1_true - c0_true

    post_samps_m0 = rand(Beta(m0.α + sumy, m0.β + n - sumy), 1_000_000)
    post_samps_m1 = rand(Beta(m1.α + sumy, m1.β + n - sumy), 1_000_000)
    ll0 = map(p -> sum(logpdf.(Bernoulli(p), y)), post_samps_m0)
    ll1 = map(p -> sum(logpdf.(Bernoulli(p), y)), post_samps_m1)

    approx_log_bf = log_bayes_factor(ll0, ll1)

    @test true_log_bf ≈ approx_log_bf atol=0.2
  end
end
