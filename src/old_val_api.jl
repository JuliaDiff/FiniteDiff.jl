
function finite_difference_derivative(f, x::T, fdtype::Type{Val{T1}},# = Val{:central}
    returntype::Type{T2}=eltype(x), f_x::Union{Nothing,T}=nothing) where {T<:Number,T1,T2}
    
    finite_difference_derivative(f, x, Val{T1}(), returntype, f_x)
end


function DerivativeCache(
    x          :: AbstractArray{<:Number},
    fx         :: Union{Nothing,AbstractArray{<:Number}},# = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}},# = nothing,
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(x)) where {T1,T2}

    DerivativeCache(x, fx, epsilon, Val{T1}(), returntype)

end


function finite_difference_derivative(
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(x),      # return type of f
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon :: Union{Nothing,AbstractArray{<:Real}} = nothing) where {T1,T2}

    finite_difference_derivative(f, x, Val{T1}(), returntype, fx, epsilon)

end




function finite_difference_derivative!(
    df         :: AbstractArray{<:Number},
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(x),
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon :: Union{Nothing,AbstractArray{<:Real}} = nothing) where {T1,T2}
    
    finite_difference_derivative!(df, f, x, Val{T1}(), returntype, fx, epsilon)

end





function GradientCache(
    df         :: Union{<:Number,AbstractArray{<:Number}},
    x          :: Union{<:Number, AbstractArray{<:Number}},
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    GradientCache(df, x, Val{T1}(), returntype, Val{T3}())

end





function GradientCache(
    c1         :: Union{Nothing,AbstractArray{<:Number}},
    c2         :: Union{Nothing,AbstractArray{<:Number}},
    fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}},# = nothing,
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(c1),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}


GradientCache(c1, c2, fx, Val{T1}(), returntype, Val{T3}())


end

function finite_difference_gradient(f, x, fdtype::Type{Val{T1}},# =Val{:central},
    returntype::Type{T2}=eltype(x), inplace::Type{Val{T3}}=Val{true},
    fx::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c1::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c2::Union{Nothing,AbstractArray{<:Number}}=nothing) where {T1,T2,T3}

    finite_difference_gradient(f, x, Val{T1}(), returntype, Val{T3}(), fx, c1, c2)

end





function finite_difference_gradient!(df, f, x, fdtype::Type{Val{T1}},# =Val{:central},
    returntype::Type{T2}=eltype(df), inplace::Type{Val{T3}}=Val{true},
    fx::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c1::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c2::Union{Nothing,AbstractArray{<:Number}}=nothing,
    ) where {T1,T2,T3}
    
    finite_difference_gradient!(df, f, x, Val{T1}(), returntype, Val{T3}(), fx, c1, c2)

end

function JacobianCache(
    x,
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    JacobianCache(x, Val{T1}(), returntype, Val{T3}())

end


function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    JacobianCache(x, fx, Val{T1}(), returntype, Val{T3}())

end


function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{Val{T1}},# = Val{:central},
    returntype :: Type{T2} = eltype(fx),
    inplace :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    JacobianCache( x1, fx, fx1, Val{T1}(), returntype, Val{T3}())

end



function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{Val{T1}},# =Val{:central},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true}) where {T1,T2,T3}

    finite_difference_jacobian(f, x, Val{T1}(), returntype, Val{T3}())

end