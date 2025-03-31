using MAT,CSV,DataFrames
using JLD2 
##
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties","FreeFlight")
##
info = CSV.read(joinpath(datadir,"MothInfoFreeFlight.csv"),DataFrame)
flower = CSV.read(joinpath(datadir,"FlowerPosFreeFlight.csv"),DataFrame)
moth = CSV.read(joinpath(datadir,"MothPosFreeFlight.csv"),DataFrame)
##
mothref = Dict{String,String}()
for i in 1:nrow(info)
    mothref[info.date[i]] = "moth_$(i)"
end
# his date format is kinda dumb, I am just going to do moths 1-12 for now 
##
FreeFlight = Dict()
## renaming flower and moth dfs 
stripped_names = (replace.(names(flower), r"run" => ""))
stripped_names = (replace.(stripped_names, r"data" => ""))
rename!(flower,stripped_names)
stripped_names = (replace.(names(moth), r"run" => ""))
stripped_names = (replace.(stripped_names, r"data" => ""))
rename!(moth,stripped_names)
##
moths = unique((replace.(stripped_names, r"_(pre|post)" => "")))

for m in moths
    FreeFlight[mothref[m]] = Dict(
        "flower_pre" => convert(Vector{Float64},flower[!,m*"_pre"]),
        "flower_post" => convert(Vector{Float64},flower[!,m*"_post"]),
        "moth_pre" => convert(Vector{Float64},moth[!,m*"_pre"]), 
        "moth_post" => convert(Vector{Float64},moth[!,m*"_post"])
        )
end
##
@save "fat_moth_free_flight.jld" FreeFlight
