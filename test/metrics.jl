@testset "metrics" begin
  x = randn(10000)
  ll = normlogpdf.(x)
  mydic = dic(ll)
end

