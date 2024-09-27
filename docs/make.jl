using IR2Reg
using Documenter
using Pkg
using TOML

DocMeta.setdocmeta!(IR2Reg, :DocTestSetup, :(using IR2Reg); recursive = true)


version = "0.1.0"

const page_rename = Dict("developer.md" => "Developer docs")
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(
    modules = [IR2Reg],
    authors = "Nathan Allaire ",
    repo = "https://github.com/nathanemac/iR2Reg.jl",
    sitename = "iR2Reg.jl",
    format = Documenter.HTML(
        edit_link = "https://github.com/nathanemac/iR2Reg.jl/edit/main/{path}",
        repolink = "https://github.com/nathanemac/iR2Reg.jl",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(
    repo = "github.com/nathanemac/iR2Reg.jl.git",
    target = "gh-pages",
    #versions = [
    #    "stable" => "v$(version)",  # for tagged versions
    #    "dev" => "main",             # dev branch
    #],
    devbranch = "main",
)
