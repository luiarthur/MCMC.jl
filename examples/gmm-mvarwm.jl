# NOTE: This is an example of how to do Mv Adaptive Metropolis for a GMM.
# Updates are done iteratively for each set of parameters, instead of jointly
# for all parameters.

import Pkg; Pkg.activate("../")
using MCMC
using BangBang
using Bijectors
using Distributions
import Random

# Set random seed for reproducibility
Random.seed!(0)

# Define model.
struct Gmm <: MCMC.Model
  y::Vector{Float64}
  K::Int
end

# Create initial state maker.
MCMC.make_init_state(m::Gmm) = (mu=randn(m.K), sigma=rand(m.K), eta=fill(1/m.K, m.K))

# Define loglikelihood and full conditionals
loglike(m, s) = sum(logpdf.(UnivariateGMM(s.mu, s.sigma, Categorical(s.eta)), m.y))
function sigma_cond(m, s, x)
  return sum(logpdf.(LogNormal(0, 1), x)) + loglike(m, setproperty!!(s, :sigma, x))
end
function mu_cond(m, s, x)
  return sum(logpdf.(Normal(0, 1), x)) + loglike(m, setproperty!!(s, :mu, x))
end
function eta_cond(m, s, x)
  return logpdf(Dirichlet(m.K, 1/m.K), x) + loglike(m, setproperty!!(s, :eta, x))
end

# Define true model.
true_model = MixtureModel(Normal.([-2, 0, 2], [.2, .2, .3]), [.3, .4, .3])

# Simulate data.
y = rand(true_model, 500)

# Instantiate model.
model = Gmm(y, 3)

# Initialize model state.
init = MCMC.make_init_state(model)

# Create sampler.
r1 = RWM(:mu, mu_cond, mvarwm(init.mu))
r2 = RWM(:sigma, sigma_cond, mvarwm(init.sigma), bijector=Bijectors.Log{1}())
r3 = RWM(:eta, eta_cond, mvarwm(init.eta), bijector=Bijectors.SimplexBijector{true}())
spl = MCMC.Gibbs(model, r1, r2, r3)

# Sample via Adaptive Metropolis.
chain = mcmc(spl, 2000, nburn=10000, thin=2).chain

# Check model fit. 
ord = sortperm(mean(getindex.(chain, :mu)))
@assert isapprox(mean(getindex.(chain, :mu))[ord], mean.(true_model.components), atol=0.1)
@assert isapprox(mean(getindex.(chain, :sigma))[ord], std.(true_model.components), atol=0.1)
@assert isapprox(mean(getindex.(chain, :eta))[ord], true_model.prior.p, atol=0.05)
