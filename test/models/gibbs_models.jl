struct M1 <: MCMC.Model end
MCMC.make_init_state(m::M1) = (mu=0.0, sigma=0)

struct M2 <: MCMC.Model end
MCMC.make_init_state(m::M2) = (mu=0.0, sigma=0, eta=zeros(3))

struct M3 <: MCMC.Model end
MCMC.make_init_state(m::M3) = (mu=0.0, sigma=0.0, eta=zeros(3))
