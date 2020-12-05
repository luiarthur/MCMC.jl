# DIC
deviance(loglikelihood::Real) = -2 * loglikelihood
function dic(loglikelihood::AbstractVector{<:Real})
  D = deviance.(loglikelihood)
  return mean(D) + 0.5 * var(D)
end
