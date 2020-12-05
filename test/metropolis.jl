@testset "metropolis" begin
  function sample_with_metropolis(n::Integer, metropolis::Function)
    x = 0.0
    xs = Float64[]
    for _ in 1:n
      x = metropolis(x, normlogpdf, 1)
      append!(xs, x)
    end
    return xs
  end
  
  Random.seed!(0)
  xs = sample_with_metropolis(1_000_000, gaussian_random_walk_metropolis)
  @test isapprox(mean(xs), 0, atol=1e-2)
  @test isapprox(std(xs), 1, atol=1e-2)
  @test isapprox(quantile(xs, 0.025), -1.96, atol=1e-2)
  @test isapprox(quantile(xs, 0.975), 1.96, atol=1e-2)
end
