"""
Needs to be overloaded.
"""
function update(model::Model, s::T) where {T} end

"""
Needs to be overloaded.
"""
function make_init_state(model::Model) end
make_init_state(gibbs::Gibbs) = make_init_state(gibbs.model)

"""
Subset a named tuple
"""
function subsetnamedtuple(state::NamedTuple, names)
  vs = map(v -> state[v], names)
  return NamedTuple{names}(vs)
end

"""
    mcmc(model::Model, nsamps::Int; init::Union{NamedTuple, Nothing}, nburn::Int=0,
         thin::Int=1, exclude::Vector{Symbol}=[], callback=nothing)

### Arguments

`model`: is a subtype of MCMC.Model.

`nsamps`: Number of samples to return.

`init`: is the initial state.

`nburn`: Number of burn-in samples to discard at the beginning.

`thin`: Factor by which to thin samples.

`exclude`: Parameters to exclude from final chain.

`callback`: Callback function of the form `callback(chain, state, sample, i, metrics, iterator)`.

**Return**: `(chain::Vector{<:NamedTuple}, metrics::Dict{Symbol, Any})` where `chain` is a Vector of `NamedTuples` of length `nsamps` and `metrics` is additional metrics that would be generated through `callback` if provided.
"""
function mcmc(model::Union{Model,Gibbs}, nsamps::Int; init::Union{Nothing, NamedTuple}=nothing,
              nburn::Int=0, thin::Int=1, exclude::Vector{Symbol}=Symbol[],
              callback=nothing, progress=true, kwargs...)
  # Create intial state if needed.
  init === nothing && (init = make_init_state(model))

  # Assert that arguments are correctly specified.
  thin >= 1 || error("`thin` should be ≥ 1")
  nburn >= 0 || error("`thin` should be ≥ 0")

  # Total number of iterations.
  niters = nsamps * thin + nburn

  # Index counter
  idx = 0

  # All parameters
  allparams = keys(init)

  # Tracked parameters
  tracked_params = Tuple(filter(s -> !(s in exclude), allparams))

  # Current state
  state = deepcopy(init)

  # Preallocate output.
  chain = [deepcopy(subsetnamedtuple(state, tracked_params)) for _ in 1:nsamps]

  # Additional metrics.
  metrics = Dict{Symbol, Any}()

  iterator = progress ? ProgressBar(1:niters) : 1:niters
  for i in iterator
    # Update current state.
    # NOTE: User needs to imlement MCMC.update(s::typeof(s))
    state = update(model, state; kwargs...)

    # Trim state to get a sample to save.
    sample = subsetnamedtuple(state, tracked_params)

    # Callback function.
    callback === nothing || callback(chain, state, sample, i, metrics, iterator)

    # Save current state.
    (i > nburn) && ((i - nburn) % thin == 0) && setindex!!(chain, sample, idx += 1)
  end

  return (chain=chain, metrics=metrics)
end
