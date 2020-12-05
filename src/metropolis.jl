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
