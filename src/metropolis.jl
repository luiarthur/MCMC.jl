# Basic zero-mean Gaussian (uni/multi-variate) Metropolis step.
struct StaticRWM{D <: Union{Normal,MvNormal}} <: Metropolis
  proposal::D
end
function _update(rng::Random.AbstractRNG, rwm::StaticRWM, curr::T, logprob::Function) where T
  cand = curr + rand(rng, rwm.proposal)
  log_acceptance_prob = logprob(cand) - logprob(curr)
  accept = log_acceptance_prob > -Random.randexp(rng)
  draw = accept ? cand : curr
  return (draw, accept)
end


# Static Random Walk Metropolis for Gibbs.
"""
`name::Symbol`

`stepper::Function`  function that takes `(model::Model, state::T) where T` and returns updated value for parameter with name `name`.

`proposal::Union{Normal,MvNormal}`  proposal distribution
"""
struct _StaticRWM{S<:OneOrMoreSymbols, F<:Function, T<:StaticRWM}
  name::S
  stepper::F
  rwm::T
end
function RWM(name::Symbol, logprob::Function, proposal::Union{Normal, MvNormal})
  srwm = StaticRWM(proposal)
  function stepper(model::Model, state::S) where S
    _logprob(x) = logprob(model, state, x)
    return update(srwm, state[name], _logprob)
  end
  return _StaticRWM(name, stepper, srwm)
end



# TODO
# - [ ] Multivariate ARWM (MvARWM)
# - [ ] Adaptive RWM (ARWM)
