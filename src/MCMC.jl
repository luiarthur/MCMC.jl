module MCMC

using BangBang
using Distributions
using ProgressBars
using StatsFuns
import Random
using Bijectors
import LinearAlgebra

abstract type Model end
abstract type InferenceAlgorithm end
const OneOrMoreSymbols = Union{Symbol, NTuple{N, Symbol} where N}

abstract type Metropolis end
# Implement _update(::Random.AbstractRNG, ::Metropolis, curr::Real, logprob::Function)
function _update(met::Metropolis, curr::T, logprob::Function) where T
  return _update(Random.GLOBAL_RNG, met, curr, logprob)
end
function update(rng::Random.AbstractRNG, met::Metropolis, curr::T, logprob::Function) where T
  return _update(rng, met, curr, logprob)[1]
end
function update(met::Metropolis, curr::T, logprob::Function) where T
  return update(Random.GLOBAL_RNG, met, curr, logprob)
end

# Utilities.
include("misc.jl")

# Metropolis.
include("static_rwm.jl")
include("rwm.jl")
include("arwm.jl")
include("mvarwm.jl")

# General stuff.
include("inference/gibbs.jl")
include("mcmc.jl")
include("metrics.jl")

export mcmc, wsample_logprob, Gibbs, Conditional
export dic, deviance, log_bayes_factor
export RWM, mvarwm
export quantiles, logmeanexp, hellinger

end # module
