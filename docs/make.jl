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
