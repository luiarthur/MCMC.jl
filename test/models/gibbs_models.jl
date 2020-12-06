struct M1 <: MCMC.Model end
MCMC.make_init_state(m::M1) = (mu=0.0, sigma=0)

struct M2 <: MCMC.Model end
MCMC.make_init_state(m::M2) = (mu=0.0, sigma=0, eta=zeros(3))

struct M3 <: MCMC.Model end
MCMC.make_init_state(m::M3) = (mu=0.0, sigma=0.0, eta=zeros(3))

struct M4 <: MCMC.Model end
MCMC.make_init_state(m::M4) = (theta=0.1, mu=.2)

struct M5 <: MCMC.Model K::Int end
MCMC.make_init_state(m::M5) = (theta=zeros(m.K), mu=.2)
