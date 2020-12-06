# Random Walk Metropolis for Gibbs.
"""
`name::Symbol`

`stepper::Function`  function that takes `(model::Model, state::T) where T` and returns updated value for parameter with name `name`.

`proposal::Union{Normal,MvNormal}`  proposal distribution
"""
struct RWM{S<:OneOrMoreSymbols, F<:Function, T<:Metropolis}
  name::S
  rwm::T
  stepper::F
end

"""
`bijector`: This should be the transformtion to the real space. e.g. if X has positive support, the Log transform should be supplied.
"""
function RWM(name::Symbol, logprob::Function, met::Metropolis; bijector=nothing)
  if bijector === nothing
    if met isa StaticRWM 
      if met.proposal isa Normal
        bijector = Bijectors.Identity{0}()
      else
        bijector = Bijectors.Identity{1}()
      end
    elseif met isa MvAdaptiveRWM
      bijector = Bijectors.Identity{1}()
    else
      error("Not implemented!")
    end
  end
  invb = inv(bijector)

  function stepper(model::Model, state::S) where S
    function _logprob(real_x)
      x, labsdj = forward(invb, real_x)
      return logprob(model, state, x) + labsdj
    end
    return invb(update(met, bijector(state[name]), _logprob))
  end

  return RWM(name, met, stepper)
end

function RWM(name::Symbol, logprob::Function, proposal::Union{Normal, MvNormal};
             bijector=nothing)
  srwm = StaticRWM(proposal)
  return RWM(name, logprob, srwm, bijector=bijector)
end

# TODO: Implement this.
# function RWM(name::Tuple, logprob::Function, proposal::Union{Normal, MvNormal};
#              bijector=nothing)
# end
