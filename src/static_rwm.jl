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



