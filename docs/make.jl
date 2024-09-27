using IR2Reg
using Documenter

DocMeta.setdocmeta!(IR2Reg, :DocTestSetup, :(using IR2Reg); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [IR2Reg],
    authors = "Nathan Allaire ",
    repo = "https://github.com/nathanemac/IR2Reg.jl/blob/{commit}{path}#{line}",
    sitename = "IR2Reg.jl",
    format = Documenter.HTML(
        edit_url = "https://github.com/nathanemac/IR2Reg.jl/edit/{branch}/{path}",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(
    repo = "github.com/nathanemac/IR2Reg.jl.git",
    target = "gh-pages",
    versions = [
        "stable" => "v$(IR2Reg.version)",  # for tagged versions
        "dev" => "main",  # dev branch
    ],
    devbranch = "main",
)
