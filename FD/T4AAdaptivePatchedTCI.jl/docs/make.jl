using T4AAdaptivePatchedTCI
using Documenter

DocMeta.setdocmeta!(
    T4AAdaptivePatchedTCI, :DocTestSetup, :(using T4AAdaptivePatchedTCI); recursive=true
)

makedocs(;
    modules=[T4AAdaptivePatchedTCI],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="T4AAdaptivePatchedTCI.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4AAdaptivePatchedTCI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/tensor4all/T4AAdaptivePatchedTCI.jl.git", devbranch="main")
