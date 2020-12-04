@testset "mcmc.jl" begin
  @testset "subsetnamedtuple" begin
    state = generate_initial_state(10)
    sample = MCMCDev.subsetnamedtuple(state, (:a, :b))
    @test Set(keys(sample)) == Set((:a, :b))
  end

  @testset "mcmc" begin
    state = generate_initial_state(10)
    nsamps = 1000
    nburn = 44
    thin = 33
    output = mcmc(MySimpleModel(), state, nsamps, nburn=nburn, thin=thin, exclude=[:c])
    chain = output.chain

    @test length(chain) == nsamps
    @test chain[end].a == nsamps * thin + nburn
    @test chain[1].a == nburn + thin
  end

  @testset "thin, nburn" begin
    state = generate_initial_state(10)
    nsamps = 1000
    nburn = 44
    thin = 33

    Random.seed!(1234)
    ref_chain = mcmc(MySimpleModel(), state, nsamps * thin + nburn,
                     exclude=[:c]).chain

    Random.seed!(1234)
    thinned_chain = mcmc(MySimpleModel(), state, nsamps, thin=thin,
                         exclude=[:c]).chain

    Random.seed!(1234)
    burned_chain = mcmc(MySimpleModel(), state, nsamps, nburn=nburn,
                        exclude=[:c]).chain

    Random.seed!(1234)
    thinned_and_burned_chain = mcmc(MySimpleModel(), state, nsamps, nburn=nburn,
                                    thin=thin, exclude=[:c]).chain

    @test length(ref_chain) == nsamps * thin + nburn
    @test length(thinned_chain) == nsamps
    @test length(burned_chain) == nsamps
    @test length(thinned_and_burned_chain) == nsamps
    @test all(thinned_chain .== ref_chain[thin:thin:nsamps*thin])
    @test all(burned_chain .== ref_chain[nburn+1:nburn+nsamps])
    @test all(thinned_and_burned_chain .== ref_chain[nburn+thin:thin:nburn+thin*nsamps])
  end
end
