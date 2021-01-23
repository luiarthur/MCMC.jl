# Gaussian mixture model
struct GMM{T<:AbstractVector{<:Real}, M<:Normal, S<:InverseGamma, W<:Dirichlet} <: MCMC.Model
  y::T
  K::Int
  mu::M
  sigmasq::S
  eta::W
end
function GMM(y::AbstractVector{<:Real}, K::Int; mu=Normal(0, 1),
             sigmasq=InverseGamma(3, 2), eta=Dirichlet(K, 1/K))
  return GMM(y, K, mu, sigmasq, eta)
end
function MCMC.make_init_state(m::GMM)
  nobs = length(m.y)
  eta = rand(m.eta)
  return (mu=rand(m.mu, m.K), sigmasq=rand(m.sigmasq, m.K),
          eta=eta, lambda=rand(Categorical(eta), nobs))
end
function MCMC.update(m::GMM, s::T) where T
  s = setproperty!!(s, :mu, update_mu(m, s))
  s = setproperty!!(s, :sigmasq, update_sigmasq(m, s))
  s = setproperty!!(s, :eta, update_eta(m, s))
  s = setproperty!!(s, :lambda, update_lambda(m, s))

  return s
end
function update_mu(m::GMM, s::T) where T
  m_mu, s_mu = params(m.mu)

  vkernels = zero.(s.mu)
  mkernels = zero.(s.mu)

  for n in eachindex(m.y)
    k = s.lambda[n]
    vkernels[k] += 1
    mkernels[k] += m.y[n]
  end

  vnew = 1 ./ (s_mu^-2 .+ vkernels ./ s.sigmasq)
  mnew = vnew .* (m_mu/(s_mu^2) .+ mkernels ./ s.sigmasq)

  return randn(m.K) .* sqrt.(vnew) + mnew
end
function update_sigmasq(m::GMM, s::T) where T
  a, b = params(m.sigmasq)
  akernel = zero.(s.sigmasq)
  bkernel = zero.(s.sigmasq)

  for n in eachindex(m.y)
    k = s.lambda[n]
    akernel[k] += 1
    bkernel[k] += (m.y[n] - s.mu[k]) ^ 2
  end

  anew = a .+ akernel / 2
  bnew = b .+ bkernel / 2
  s.sigmasq .= rand.(InverseGamma.(anew, bnew))

  return s.sigmasq
end
function update_eta(m::GMM, s::T) where T
  anew = copy(collect(m.eta.alpha))
  for lam in s.lambda
    anew[lam] += 1
  end
  return rand(Dirichlet(anew))
end
function update_lambda(m::GMM, s::T) where T
  logeta = s.eta
  lambda = [let
              logmix = normlogpdf.(s.mu, sqrt.(s.sigmasq), yn) + logeta
              wsample_logprob(logmix)
            end for yn in m.y]
  return lambda
end
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
