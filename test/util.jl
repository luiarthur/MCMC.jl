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

# Gaussian mixture model
struct GMM{T<:AbstractVector{<:Real}, M<:Normal, S<:InverseGamma, W<:Dirichlet} <: MCMC.Model
  y::T
  K::Int
  mu::M
  sigmasq::S
  eta::W
end
function GMM(y::AbstractVector{<:Real}, K::Int; mu=Normal(0, 1),
             sigmasq=InverseGamma(3, 2), eta=Dirichlet(K, 1/K))
  return GMM(y, K, mu, sigmasq, eta)
end
function MCMC.make_init_state(m::GMM)
  nobs = length(m.y)
  eta = rand(m.eta)
  return (mu=rand(m.mu, m.K), sigmasq=rand(m.sigmasq, m.K),
          eta=eta, lambda=rand(Categorical(eta), nobs))
end
function MCMC.update(m::GMM, s::T) where T
  s = setproperty!!(s, :mu, update_mu(m, s))
  s = setproperty!!(s, :sigmasq, update_sigmasq(m, s))
  s = setproperty!!(s, :eta, update_eta(m, s))
  s = setproperty!!(s, :lambda, update_lambda(m, s))

  return s
end
function update_mu(m::GMM, s::T) where T
  m_mu, s_mu = params(m.mu)

  vkernels = zero.(s.mu)
  mkernels = zero.(s.mu)

  for n in eachindex(m.y)
    k = s.lambda[n]
    vkernels[k] += 1
    mkernels[k] += m.y[n]
  end

  vnew = 1 ./ (s_mu^-2 .+ vkernels ./ s.sigmasq)
  mnew = vnew .* (m_mu/(s_mu^2) .+ mkernels ./ s.sigmasq)

  return randn(m.K) .* sqrt.(vnew) + mnew
end
function update_sigmasq(m::GMM, s::T) where T
  a, b = params(m.sigmasq)
  akernel = zero.(s.sigmasq)
  bkernel = zero.(s.sigmasq)

  for n in eachindex(m.y)
    k = s.lambda[n]
    akernel[k] += 1
    bkernel[k] += (m.y[n] - s.mu[k]) ^ 2
  end

  anew = a .+ akernel / 2
  bnew = b .+ bkernel / 2
  s.sigmasq .= rand.(InverseGamma.(anew, bnew))

  return s.sigmasq
end
function update_eta(m::GMM, s::T) where T
  anew = copy(m.eta.alpha)
  for lam in s.lambda
    anew[lam] += 1
  end
  return rand(Dirichlet(anew))
end
function update_lambda(m::GMM, s::T) where T
  logeta = s.eta
  lambda = [let
              logmix = normlogpdf.(s.mu, sqrt.(s.sigmasq), yn) + logeta
              wsample_logprob(logmix)
            end for yn in m.y]
  return lambda
end
