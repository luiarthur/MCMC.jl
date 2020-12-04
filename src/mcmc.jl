"""
Needs to be overloaded.
"""
function update end

function trimstate(state, tracked_params)
  vs = map(v -> state[v], tracked_params)
  return NamedTuple{tracked_params}(vs)
end

function mcmc(initial_state::NamedTuple, nsamps::Int;
              nburn::Int=0, thin::Int=1, exclude::Vector{Symbol}=[],
              callback=nothing)
  # Assert that arguments are correctly specified.
  thin >= 1 || error("`thin` should be ≥ 1")
  nburn >= 0 || error("`thin` should be ≥ 0")

  # Total number of iterations.
  niters = nsamps * thin + nburn

  # Index counter
  idx = 0

  # All parameters
  allparams = keys(initial_state)

  # Tracked parameters
  tracked_params = Tuple(filter(s -> !(s in exclude), allparams))

  # Current state
  state = deepcopy(initial_state)

  # Preallocate output.
  chain = [deepcopy(trimstate(state, tracked_params)) for _ in 1:nsamps]

  # TODO: Implement metrics.
  metrics = ()

  for i in ProgressBar(1:niters)
    # Update current state.
    state = update(state)  # User needs to imlement MCMCDev.update(s::typeof(s))

    # Trim state to get a sample to save.
    sample = trimstate(state, tracked_params)

    # TODO: implement callback
    callback === nothing || (_ = nothing)

    # Save current state.
    (i > nburn) && ((i - nburn) % thin == 0) && setindex!!(chain, sample, idx += 1)
  end

  return (chain=chain, metrics=metrics)
end
