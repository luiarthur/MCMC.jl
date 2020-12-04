module MCMCDev

using BangBang
using Distributions
using ProgressBars
using StatsFuns

include("mcmc.jl")
include("inference/InferenceAlgorithm.jl")
include("inference/Gibbs.jl")
include("inference/ConjugateConditional.jl")

export mcmc

end # module
