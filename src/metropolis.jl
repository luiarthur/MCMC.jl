# Basic Gaussian univariate Metropolis step.
struct UniRWM{R <: Real} <: Metropolis
  proposal_sd::R
end
function _update(rng::Random.AbstractRNG, rwm::UniRWM, curr::Real, logprob::Function)
  cand = rand(rng, Normal(curr, rwm.proposal_sd))
  log_acceptance_prob = logprob(cand) - logprob(curr)
  accept = log_acceptance_prob > -Random.randexp(rng)
  draw = accept ? cand : curr
  return (draw, accept)
end

# Univariate Random Walk Metropolis.
"""
`name::Symbol`

`stepper::Function`  function that takes `(model::Model, state::T) where T` and returns updated value for parameter with name `name`.

`proposal_sd::Real`  proposal standard deviationa (Gaussian error).
"""
struct RandomWalkMetropolis{S<:OneOrMoreSymbols, F<:Function, U<:UniRWM}
  name::S
  stepper::F
  unirwm::U
end
function RWM(name::Symbol, logprob::Function, proposal_sd::Real)
  unirwm = UniRWM(proposal_sd)

  function stepper(model::Model, state::S) where S
    _logprob(x) = logprob(model, state, x)
    return update(unirwm, state[name], _logprob)
  end
  return RandomWalkMetropolis(name, stepper, unirwm)
end



# Multivariate Random Walk Metropolis.


# TODO
# - [ ] Multivariate RWM (MvRWM)
# - [ ] Adaptive RWM (ARWM)
# - [ ] Multivariate ARWM (MvARWM)
