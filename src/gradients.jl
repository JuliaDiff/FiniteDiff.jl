struct GradientCache{CacheType,CacheType2,CacheType3,fdtype,RealOrComplex}
    x1::CacheType
    fx::CacheType2
    fx1::CacheType3
end

function GradientCache(x1,fx,_fx1,fdtype::DataType=Val{:central},
                       RealOrComplex::DataType =
                       fdtype==Val{:complex} ? Val{:Real} : eltype(x1) <: Complex ?
                       Val{:Complex} : Val{:Real})
    if fdtype == Val{:complex} && _fx1 != nothing
        warn("fx1 cache is ignored when fdtype == Val{:complex}.")
        fx1 = nothing
    else
        fx1 = _fx1
    end
    GradientCache{typeof(x1),typeof(fx),typeof(fx1),
                  fdtype,RealOrComplex}(x1,fx,fx1)
end

function finite_difference_gradient(f,x,fdtype=Val{:central},
                    RealOrComplex::DataType =
                    fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ?
                    Val{:Complex} : Val{:Real})
    x1 = similar(x)
    fx = similar(x)
    fx1 = similar(x)
    cache = GradientCache(x1,fx,fx1,fdtype,RealOrComplex)
    finite_difference_gradient(f,x,cache)
end

function finite_difference_gradient(f,x,cache::GradientCache)
    G = zeros(eltype(x), length(x), length(x))
    finite_difference_gradient!(J,f,x,cache)
    J
end
