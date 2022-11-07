using Documenter, FiniteDiff

include("pages.jl")

makedocs(sitename = "FiniteDiff.jl",
         authors = "Chris Rackauckas",
         modules = [FiniteDiff],
         clean = true,
         doctest = false,
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/FiniteDiff/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/FiniteDiff.jl.git"; push_preview = true)
