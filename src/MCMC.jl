module MCMC

using BangBang
using Distributions
using ProgressBars
using StatsFuns
import Random

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

include("temporary/dirac.jl")
include("misc.jl")
include("metropolis.jl")
include("inference/gibbs.jl")
include("mcmc.jl")
include("metrics.jl")

export mcmc, wsample_logprob, Gibbs, Conditional
export gaussian_random_walk_metropolis, gaussian_random_walk_metropolis_base
export dic, deviance
export RWM

end # module
