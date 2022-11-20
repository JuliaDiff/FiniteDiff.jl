using Documenter, FiniteDiff

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "FiniteDiff.jl",
         authors = "Chris Rackauckas",
         modules = [FiniteDiff],
         clean = true,
         doctest = false,
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/FiniteDiff/stable/"),
         pages = pages)

deploydocs(repo = "github.com/JuliaDiff/FiniteDiff.jl.git"; push_preview = true)
