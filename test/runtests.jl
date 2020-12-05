using MCMC
using Test

using BangBang
using Distributions
using StatsFuns
import Random

@testset "MCMCDec" begin
  include("models/simple_model.jl")
  include("mcmc.jl")
  include("misc.jl")
  include("models/gmm.jl")
end
