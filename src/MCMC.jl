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
