# Basic univariate Metropolis step.
function gaussian_random_walk_metropolis_base(rng::Random.AbstractRNG, curr::Real,
                                              logprob::Function, proposal_sd::Real)
  cand = curr + randn(rng) * proposal_sd
  log_acceptance_prob = logprob(cand) - logprob(curr)
  accept = log_acceptance_prob > -Random.randexp(rng)
  draw = accept ? cand : curr
  return (draw, accept)
end
function gaussian_random_walk_metropolis_base(curr::Real, logprob::Function, proposal_sd::Real)
  return gaussian_random_walk_metropolis_base(Random.GLOBAL_RNG, curr, logprob, proposal_sd)
end
function gaussian_random_walk_metropolis(rng::Random.AbstractRNG, curr::Real,
                                         logprob::Function, proposal_sd::Real)
  return gaussian_random_walk_metropolis_base(rng, curr, logprob, proposal_sd)[1]
end
function gaussian_random_walk_metropolis(curr::Real, logprob::Function, proposal_sd::Real)
  return gaussian_random_walk_metropolis(Random.GLOBAL_RNG, curr, logprob, proposal_sd)
end

# Univariate Random Walk Metropolis.
struct RandomWalkMetropolis{S<:OneOrMoreSymbols, F<:Function, R<:Real}
  name::S
  stepper::F  # function that takes `(model::Model, state::T) where T` and
              # returns updated value for parameter with name `name`.
  proposal_sd::R
end
function RWM(name::Symbol, logprob::Function, proposal_sd::Real)
  function stepper(model::Model, state::S) where S
    _logprob(x) = logprob(model, state, x)
    return gaussian_random_walk_metropolis(state[name], _logprob, proposal_sd)
  end
  return RandomWalkMetropolis(name, stepper, proposal_sd)
end

# Adaptive Random Walk Metropolis.

# TODO
# - [ ] Adaptive RWM (ARWM)
# - [ ] Multivariate RWM (MvRWM)
# - [ ] Multivariate ARWM (MvARWM)
