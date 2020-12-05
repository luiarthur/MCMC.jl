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
    output = mcmc(MySimpleModel(10), nsamps, init=state, nburn=nburn, thin=thin)
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
    ref_chain = mcmc(model, nsamps * thin + nburn, init=state, exclude=[:c]).chain

    Random.seed!(1234)
    thinned_chain = mcmc(model, nsamps, init=state, thin=thin,
                         exclude=[:c]).chain

    Random.seed!(1234)
    burned_chain = mcmc(model, nsamps, init=state, nburn=nburn,
                        exclude=[:c]).chain

    Random.seed!(1234)
    thinned_and_burned_chain = mcmc(model, nsamps, init=state, nburn=nburn,
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
    ref_chain = mcmc(model, 20, init=init).chain

    Random.seed!(1234)
    uninitialized_chain = mcmc(model, 20).chain

    @test all(ref_chain .== uninitialized_chain)
  end

  @testset "gibbs" begin
    spl = Gibbs(M1(),
                Conditional(:mu, (m, s) -> s.mu + 1),
                Conditional(:sigma, (m, s) -> s.sigma + 2))
    nsamps = 1000
    chain = mcmc(spl, nsamps).chain
    @test chain[end].mu == nsamps
    @test chain[end].sigma == nsamps * 2
  end

  @testset "gibbs multi-step" begin
    spl = Gibbs(M2(),
                Conditional((:mu, :sigma), (m, s) -> (mu = s.mu + 1, sigma = s.sigma + 2)),
                Conditional(:eta, (m, s) -> s.eta .- 1))
    nsamps = 1000
    chain = mcmc(spl, nsamps).chain
    @test chain[end].mu == nsamps
    @test chain[end].sigma == nsamps * 2
    @test all(chain[end].eta .== -nsamps)
  end

  # TODO: Add tests for callback, metrics.
end
