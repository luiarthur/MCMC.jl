# Keep this until Distributions.jl merges #1231.
# See: https://github.com/JuliaStats/Distributions.jl/pull/1231
"""
    Dirac(x)

A *Dirac distribution* is parameterized by its only value `x`, and takes its value with probability 1.

```math
P(X = \\hat{x}) = \\begin{cases}
1 & \\quad \\text{for } \\hat{x} = x, \\\\
0 & \\quad \\text{for } \\hat{x} \\neq x.
\\end{cases}
```

```julia
Dirac(2.5)   # Dirac distribution with value x = 2.5
```

External links:

* [Dirac measure on Wikipedia](http://en.wikipedia.org/wiki/Dirac_measure)
"""
struct Dirac{T} <: DiscreteUnivariateDistribution
    value::T
end

Base.eltype(::Type{Dirac{T}}) where {T} = T

Distributions.insupport(d::Dirac, x::Real) = x == d.value
Distributions.minimum(d::Dirac) = d.value
Distributions.maximum(d::Dirac) = d.value

#### Properties
Distributions.mean(d::Dirac) = d.value
Distributions.var(d::Dirac) = 0.0

Distributions.mode(d::Dirac) = d.value

Distributions.entropy(d::Dirac) = 0.0

#### Evaluation
Distributions.pdf(d::Dirac, x::Real) = insupport(d, x) ? 1.0 : 0.0
Distributions.logpdf(d::Dirac, x::Real) = insupport(d, x) ? 0.0 : -Inf

Distributions.cdf(d::Dirac, x::Real) = x < d.value ? 0.0 : 1.0

Distributions.quantile(d::Dirac{T}, p::Real) where {T} = 0 <= p <= 1 ? d.value : T(NaN)

Distributions.mgf(d::Dirac, t) = exp(t * d.value)
Distributions.cf(d::Dirac, t) = cis(t * d.value)

#### Sampling

Distributions.rand(rng::Random.AbstractRNG, d::Dirac) = d.value
