using MCMC
using Test

using BangBang
using Distributions
using StatsFuns
import Random

@testset "MCMC" begin
  include("models/simple_model.jl")
  include("models/gibbs_models.jl")
  include("mcmc.jl")
  include("misc.jl")
  include("models/gmm.jl")
end
