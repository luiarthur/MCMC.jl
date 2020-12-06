# Adaptive MvNormal random walk.
# See: https://www.tandfonline.com/doi/abs/10.1198/jcgs.2009.06134
mutable struct MvAdaptiveRWM{M <: AbstractVector{<:Real}, V <: AbstractMatrix{<:Real},
                             B <: Real} <: Metropolis
  sample_mean::M
  sample_cov::V
  beta::B
  iter::Int
  d::Int
end
function MvAdaptiveRWM(v::AbstractVector{<:Real}; beta=0.05, iter=1)
  d = length(v)
  return MvAdaptiveRWM(v, eye(d) * 0.01 / d, beta, iter, d)
end
function update_stats!(rwm::MvAdaptiveRWM, x::AbstractVector{<:Real})
  rwm.iter += 1
  update_mean!(rwm.sample_mean, x, rwm.iter)
  update_cov!(rwm.sample_cov, rwm.sample_mean, x, rwm.iter)
end
function _update(rng::Random.AbstractRNG, rwm::MvAdaptiveRWM, curr::AbstractVector{<:Real},
                 logprob::Function)
  # Update summary stats.
  update_stats!(rwm, curr)

  # Construct proposal covariance.
  proposal_cov = if rwm.iter <= 2 * rwm.d || rwm.beta > rand()
    0.01 * eye(rwm.d) / rwm.d
  else
    5.6644 * rwm.sample_cov / rwm.d
  end
  proposal_cov .= Matrix(LinearAlgebra.Symmetric(proposal_cov))

  return _update(rng, StaticRWM(MvNormal(proposal_cov)), curr, logprob)
end
const mvarmw = MvAdaptiveRWM
