@testset "metropolis" begin
  @testset "gaussian_random_walk_metropolis" begin
    function sample_with_metropolis(n::Integer, rwm::MCMC.Metropolis)
      x = 0.0
      xs = Float64[]
      for _ in 1:n
        x = MCMC.update(rwm, x, normlogpdf)
        append!(xs, x)
      end
      return xs
    end
    
    Random.seed!(0)
    xs = sample_with_metropolis(1_000_000, MCMC.StaticRWM(Normal()))
    @test isapprox(mean(xs), 0, atol=1e-2)
    @test isapprox(std(xs), 1, atol=1e-2)
    @test isapprox(quantile(xs, 0.025), -1.96, atol=1e-2)
    @test isapprox(quantile(xs, 0.975), 1.96, atol=1e-2)
  end

  @testset "RWM univariate" begin
    mu1, sd1 = (0, 1)
    mu2, sd2 = (5, .5)
    d1 = Normal(mu1, sd1)
    d2 = Normal(mu2, sd2)

    spl = Gibbs(M3(),
                RWM(:mu, (m, s, x) -> normlogpdf(mu1, sd1, x), Normal(0, 1)),
                RWM(:sigma, (m, s, x) -> normlogpdf(mu2, sd2, x), Normal(0, 1)),
                Conditional(:eta, (m, s) -> s.eta .- 1))

    nsamps = 10000
    Random.seed!(0)
    chain = mcmc(spl, nsamps, discard=500, thin=2).chain
    mu = getindex.(chain, :mu)
    sigma = getindex.(chain, :sigma)

    for (param, d) in zip([mu, sigma], [d1, d2])
      @test isapprox(mean(param), mean(d), atol=0.05)
      @test isapprox(std(param), std(d), atol=0.05)
      @test isapprox(quantile(param, 0.025), quantile(d, .025), atol=0.05)
      @test isapprox(quantile(param, 0.975), quantile(d, .975), atol=0.05)
    end
  end

  @testset "RWM multivariate" begin
    mu1, sd1 = (0, 1)
    mu2, sd2 = (5, .5)
    d1 = Normal(mu1, sd1)
    d2 = Normal(mu2, sd2)
    d3 = MvNormal(0:2, 1)

    spl = Gibbs(M3(),
                RWM(:mu, (m, s, x) -> normlogpdf(mu1, sd1, x), Normal(0, 1)),
                RWM(:sigma, (m, s, x) -> normlogpdf(mu2, sd2, x), Normal(0, 1)),
                RWM(:eta, (m, s, x) -> logpdf(d3, x), MvNormal(3, 0.1/3)))

    nsamps = 50000
    Random.seed!(1)
    chain = mcmc(spl, nsamps, discard=10000, thin=2).chain
    mu = getindex.(chain, :mu)
    sigma = getindex.(chain, :sigma)

    for (param, d) in zip([mu, sigma], [d1, d2])
      @test isapprox(mean(param), mean(d), atol=0.05)
      @test isapprox(std(param), std(d), atol=0.05)
      @test isapprox(quantile(param, 0.025), quantile(d, .025), atol=0.05)
      @test isapprox(quantile(param, 0.975), quantile(d, .975), atol=0.05)
    end

    eta = getindex.(chain, :eta)
    @test isapprox(mean(eta), mean(d3), atol=0.2)
    @test isapprox(std(eta), LinearAlgebra.diag(d3.Σ), atol=0.2)
  end
end
