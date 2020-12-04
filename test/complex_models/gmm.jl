@testset "GMM" begin
  true_model = MixtureModel([Normal(3, .6), Normal(-2, 1)], [.3, .7])

  Random.seed!(0)

  nobs = 1000
  y = rand(true_model, nobs)
  model = GMM(y, 2)
  nsamps = 1000
  chain, _ = mcmc(model, nsamps, nburn=1000)

  mu = hcat(map(c -> c.mu, chain)...)
  sigma = hcat(map(c -> sqrt.(c.sigmasq), chain)...)
  eta = hcat(map(c -> c.eta, chain)...)
  lambda = hcat(map(c -> c.lambda, chain)...)

  @test size(mu) == (model.K, nsamps)
  @test size(sigma) == (model.K, nsamps)
  @test size(eta) == (model.K, nsamps)
  @test size(lambda) == (nobs, nsamps)

  function match_components(postmean::AbstractVector{<:Real}, truth::AbstractVector{<:Real}, atol)
    cond1 = abs2(postmean[1] - truth[1]) < atol
    cond2 = abs2(postmean[2] - truth[2]) < atol
    cond3 = abs2(postmean[1] - truth[2]) < atol
    cond4 = abs2(postmean[2] - truth[1]) < atol
    return (cond1 && cond2) || (cond3 && cond4)
  end
  function match_components(postmean, sym::Symbol, atol)
    truth = [getfield(true_model.components[1], sym),
             getfield(true_model.components[2], sym)]
    return match_components(postmean, truth, atol)
  end
  mean_mu = vec(mean(mu, dims=2))
  mean_sigma = vec(mean(sigma, dims=2))
  mean_eta = vec(mean(eta, dims=2))
  # println(mean_mu)
  # println(mean_sigma)
  # println(mean_eta)
  @test match_components(mean_mu, :μ, 1e-1)
  @test match_components(mean_sigma, :σ, 1e-2)
  @test match_components(mean_eta, true_model.prior.p, 1e-2)
end
