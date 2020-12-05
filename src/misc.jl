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
