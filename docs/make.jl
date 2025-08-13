using Documenter, FiniteDiff

DocMeta.setdocmeta!(
    FiniteDiff,
    :DocTestSetup,
    :(using FiniteDiff);
    recursive=true,
)

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force=true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force=true)

# create index from README and contributing
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/JuliaDiff/FiniteDiff.jl/blob/master/README.md"
        ```
        """,
    )
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
    
    # Add API reference content
    println(io, "")
    println(io, "## API Reference")
    println(io, "")
    println(io, "```@docs")
    println(io, "FiniteDiff")
    println(io, "```")
    println(io, "")
    println(io, "FiniteDiff.jl provides fast, non-allocating finite difference calculations with support for sparsity patterns and various array types. The API is organized into several categories:")
    println(io, "")
    println(io, "### Function Categories")
    println(io, "")
    println(io, "- **[Derivatives](@ref derivatives)**: Single and multi-point derivatives of scalar functions")
    println(io, "- **[Gradients](@ref gradients)**: Gradients of scalar-valued functions with respect to vector inputs")
    println(io, "- **[Jacobians](@ref jacobians)**: Jacobian matrices of vector-valued functions, including sparse Jacobian support")
    println(io, "- **[Hessians](@ref hessians)**: Hessian matrices of scalar-valued functions")
    println(io, "- **[Jacobian-Vector Products](@ref jvp)**: Efficient computation of directional derivatives without forming full Jacobians")
    println(io, "- **[Utilities](@ref utilities)**: Internal utilities and helper functions")
    println(io, "")
    println(io, "### Quick Start")
    println(io, "")
    println(io, "All functions follow a consistent API pattern:")
    println(io, "")
    println(io, "- **Cache-less versions**: `finite_difference_*` - convenient but allocate temporary arrays")
    println(io, "- **In-place versions**: `finite_difference_*!` - efficient, non-allocating when used with caches")
    println(io, "- **Cache constructors**: `*Cache` - pre-allocate work arrays for repeated computations")
    println(io, "")
    println(io, "### Method Selection")
    println(io, "")
    println(io, "Choose your finite difference method based on accuracy and performance needs:")
    println(io, "")
    println(io, "- **Forward differences**: Fast, `O(h)` accuracy, requires `O(n)` function evaluations")
    println(io, "- **Central differences**: More accurate `O(h²)`, requires `O(2n)` function evaluations")
    println(io, "- **Complex step**: Machine precision accuracy, `O(n)` evaluations, requires complex-analytic functions")
    println(io, "")
    println(io, "### Performance Tips")
    println(io, "")
    println(io, "1. **Use caches** for repeated computations to avoid allocations")
    println(io, "2. **Consider sparsity** for large Jacobians with known sparsity patterns")
    println(io, "3. **Choose appropriate methods** based on your accuracy requirements")
    println(io, "4. **Leverage JVPs** when you only need directional derivatives")
    println(io, "")
    
    for line in eachline(joinpath(@__DIR__, "src", "reproducibility.md"))
        println(io, line)
    end
end

include("pages.jl")

makedocs(sitename="FiniteDiff.jl",
    authors="Chris Rackauckas",
    modules=[FiniteDiff],
    clean=true,
    doctest=false,
    format=Documenter.HTML(assets=["assets/favicon.ico"],
        canonical="https://docs.sciml.ai/FiniteDiff/stable/"),
    pages=pages)

deploydocs(repo="github.com/JuliaDiff/FiniteDiff.jl.git"; push_preview=true)
