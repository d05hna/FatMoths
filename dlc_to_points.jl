using CSV
using DataFrames
using SavitzkyGolay
using Glob
using GLMakie
using Printf
using JLD2
using HDF5
## get the files
datadir ="/media/doshna/PutSponDosh/DOSHNA/FatMoths/Videos/Flower"

files = sort(getindex.(split.(Glob.glob("*.csv",datadir),"/"), 9 ))

dates = unique(getindex.(split.(files,"_"),1))

## add the 
alldata = Dict()
for d in dates
    tmps = strip.(sort(filter(x -> occursin(d,x),files)))
    nam = strip.(getindex.(split.(tmps,"DLC"),1))    
    n_trials = Int(length(tmps)/3)
    trials = vcat([fill(@sprintf("%03d", i - 1), 3) for i in 1:n_trials]...)
    new_tmps = [
        string(split(t, "_")[1], "_", trials[i], "_", split(t, "_")[3])
        for (i, t) in enumerate(nam)
        ]

    for (i,f) in enumerate(tmps) 
        colnames = Symbol.(collect(CSV.read(joinpath(datadir,f),DataFrame)[2,:]))
        df = CSV.read(joinpath(datadir,f),DataFrame;skipto=4,header=colnames)
        alldata[new_tmps[i]] = df
    end
end

##
h5open("/home/doshna/Documents/PHD/data/fatties/DLC/raw_mats.h5", "w") do file
    for (key,df) in alldata
        file[key] = Matrix(df)
    end
    file["names"] = ["frame","x","y","likelihood"]
end
##

sub = filter(x->occursin(n,x), collect(keys(alldata)))
together = DataFrame()
for s in sort(sub)
    together = vcat(together,alldata[s],cols=:union)
end
