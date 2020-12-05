module MCMC

using BangBang
using Distributions
using ProgressBars
using StatsFuns

abstract type Model end
abstract type InferenceAlgorithm end

include("inference/gibbs.jl")
include("misc.jl")
include("mcmc.jl")

export mcmc, wsample_logprob, Gibbs, Conditional

end # module
