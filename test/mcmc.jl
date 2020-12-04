@testset "mcmc.jl" begin
  @testset "trimstate" begin
    state = generate_initial_state(10)
    sample = MCMCDev.trimstate(state, (:a, :b))
    @test Set(keys(sample)) == Set((:a, :b))
  end

  @testset "mcmc" begin
    state = generate_initial_state(10)
    function MCMCDev.update(s::typeof(state))
      newstate = (a=s.a + 1, b=s.b - 1, c = s.c + randn(length(s.c)))
      return setproperties!!(s, newstate)
    end

    nsamps = 10
    nburn = 5
    thin = 2
    output = mcmc(state, nsamps, nburn=nburn, thin=thin, exclude=[:c])
    chain = output.chain

    @test length(chain) == nsamps
    @test chain[end].a == nsamps * thin + nburn
    @test chain[end].b == -(nsamps * thin + nburn)
  end

  @testset "thin, nbutn" begin
    # TODO: Test with same random seeds.
  end
end
