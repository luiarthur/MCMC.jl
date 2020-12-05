# A simple model
struct MySimpleModel <: MCMC.Model
  K::Int
end
MCMC.make_init_state(m::MySimpleModel) = (a=0, b=0.0, c=zeros(m.K))
function MCMC.update(::MySimpleModel, s::T) where T
  newstate = (a=s.a + 1,
              b=s.b + randn(),
              c=s.c .+ 1)
  return setproperties!!(s, newstate)
end
