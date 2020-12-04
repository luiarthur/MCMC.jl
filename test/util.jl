struct MySimpleModel <: MCMCDev.Model end
generate_initial_state(K) = (a=0, b=0.0, c=randn(K))
function MCMCDev.update(::MySimpleModel, s::T) where T
  newstate = (a=s.a + 1,
              b=s.b + randn(),
              c=s.c + randn(length(s.c)))
  return setproperties!!(s, newstate)
end

