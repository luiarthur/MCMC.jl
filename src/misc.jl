"""
weighted sampling: takes (unnormalized) log probs and returns index
"""
function wsample_logprob(logprobs::AbstractVector{<:Real})
  logdenom = logsumexp(logprobs)
  p = exp.(logprobs .- logdenom)
  return Distributions.wsample(p)
end
function wsample_logprob(logprobs::AbstractVector{<:Real}, n::Int)
  return [wsample_logprob(logprobs) for _ in 1:n]
end

Random.rand(rng::Random.AbstractRNG, d::Array{T}) where {T <: Distribution} = rand.(d)

function update_mean!(current_mean, x, iter)
  current_mean .+= (x - current_mean) / iter
end


function update_cov!(current_cov, current_mean, x, iter)
  d = x - current_mean
  current_cov .= current_cov * (iter - 1)/iter + (d*d') * (iter - 1)/iter^2
  current_cov .= Matrix(LinearAlgebra.Symmetric(current_cov))
end


eye(T::Type, n::Integer) = Matrix{T}(LinearAlgebra.I(n))
eye(n::Integer) = eye(Float64, n)
