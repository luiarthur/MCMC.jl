@testset "mcmc.jl" begin
  @testset "subsetnamedtuple" begin
    state = MCMC.make_init_state(MySimpleModel(10))
    sample = MCMC.subsetnamedtuple(state, (:a, :b))
    @test Set(keys(sample)) == Set((:a, :b))
  end

  @testset "mcmc" begin
    state = MCMC.make_init_state(MySimpleModel(10))
    nsamps = 1000
    nburn = 44
    thin = 33
    output = mcmc(MySimpleModel(10), state, nsamps, nburn=nburn, thin=thin)
    chain = output.chain

    @test length(chain) == nsamps
    @test chain[end].a == nsamps * thin + nburn
    @test chain[1].a == nburn + thin
    @test all(chain[end].c .== nsamps * thin + nburn)
    @test all(chain[1].c .== nburn + thin)
  end

  @testset "thin, nburn" begin
    model = MySimpleModel(10)
    state = MCMC.make_init_state(model)
    nsamps = 1000
    nburn = 44
    thin = 33

    Random.seed!(1234)
    ref_chain = mcmc(model, state, nsamps * thin + nburn,
                     exclude=[:c]).chain

    Random.seed!(1234)
    thinned_chain = mcmc(model, state, nsamps, thin=thin,
                         exclude=[:c]).chain

    Random.seed!(1234)
    burned_chain = mcmc(model, state, nsamps, nburn=nburn,
                        exclude=[:c]).chain

    Random.seed!(1234)
    thinned_and_burned_chain = mcmc(model, state, nsamps, nburn=nburn,
                                    thin=thin, exclude=[:c]).chain

    @test length(ref_chain) == nsamps * thin + nburn
    @test length(thinned_chain) == nsamps
    @test length(burned_chain) == nsamps
    @test length(thinned_and_burned_chain) == nsamps
    @test all(thinned_chain .== ref_chain[thin:thin:nsamps*thin])
    @test all(burned_chain .== ref_chain[nburn+1:nburn+nsamps])
    @test all(thinned_and_burned_chain .== ref_chain[nburn+thin:thin:nburn+thin*nsamps])
  end

  @testset "make_init_state" begin
    model = MySimpleModel(10)
    init = MCMC.make_init_state(model)

    Random.seed!(1234)
    ref_chain = mcmc(model, init, 20).chain

    Random.seed!(1234)
    uninitialized_chain = mcmc(model, 20).chain

    @test all(ref_chain .== uninitialized_chain)
  end

  # TODO: Add tests for callback, metrics.
end
