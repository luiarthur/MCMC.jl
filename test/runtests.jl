using MCMC
using Test

using BangBang
using Distributions
using StatsFuns
import Random

@testset "MCMCDec" begin
  include("util.jl")
  include("mcmc.jl")
  include("misc.jl")
  include("complex_models/gmm.jl")
end
