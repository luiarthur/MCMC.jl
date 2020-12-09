# DIC
deviance(loglikelihood::Real) = -2 * loglikelihood
function dic(loglikelihood::AbstractVector{<:Real})
  D = deviance.(loglikelihood)
  return mean(D) + 0.5 * var(D)
end

"""
log(1 / mean(exp(x)))
"""
logharmmeanexp(x::AbstractVector{<:Real}) = log(length(x)) - logsumexp(-x)

"""
  log_bayes_factor(lls::Vector{AbstractVector{<:Real}})

Compute log bayes factor for all models with respect to first model.
"""
function log_bayes_factor(lls::Vector{<:AbstractVector{<:Real}})
  harmean_of_loglike = map(logharmmeanexp, lls)

  # Ratio of harmonic means.
  return harmean_of_loglike .- harmean_of_loglike[1]
end

"""
    log_bayes_factor(ll1::AbstractVector{<:Real}, ll2::AbstractVector{<:Real})

Returns: log bayes factor in favor of model2 (vs. model1).

# Arguments

`ll1`: loglikelihood evaluations for model 1 for each posterior sample.
`ll2`: loglikelihood evaluations for model 2 for each posterior sample.
"""
function log_bayes_factor(ll1::AbstractVector{<:Real}, ll2::AbstractVector{<:Real})
  return log_bayes_factor([ll1, ll2])[2]
end
