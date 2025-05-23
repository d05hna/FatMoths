using CSV
using DataFrames
using HDF5
using Glob
using Pipe
using DataFramesMeta
using GLMakie
using DSP 
using FFTW
using JLD2
using SavitzkyGolay
include("me_functions.jl")
##
@load "fat_moths_set_1.jld" allmoths
datadir = "/home/doshna/Documents/PHD/data/fatties/"

moth = "2025_01_30"
##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

mothstarts = Dict(
    "2024_11_01" => [Int(1.5e5),Int(1)],
    "2024_11_04" => [Int(1.5e5),Int(1)],
    "2024_11_05" => [Int(1),Int(1e5)],
    "2024_11_07" => [Int(1e4),Int(1)],
    "2024_11_08" => [Int(1e5),Int(1)],
    "2024_11_11" => [Int(1),Int(1)],
    "2024_11_20" => [Int(1e5),Int(1.5e5)],
    "2024_12_03" => [Int(1e5),Int(1.5e5)],
    "2025_01_30" => [Int(1e4),Int(1.25e5)]


)
##
function get_stims(moth,datadir,mothstarts)
    fs = 1e4
    camfs = 300
    paths = glob("*$moth*.csv",joinpath(datadir,"DLTData"))

    tris = sort(unique([split(split(p,"/")[end],"_")[4] for p in paths]))
    println("for moth $moth tris are $tris")
    ##

    pres = filter(s -> occursin("$(moth)_$(tris[1])", s), paths)
    posts = filter(s->occursin("$(moth)_$(tris[2])",s),paths)

    ##
    prepos = []
    for p in pres
        d = CSV.read(p,DataFrame)
        push!(prepos,d.pt1_cam1_X...)
    end

    pospos = []
    for p in posts 
        d = CSV.read(p,DataFrame)
        push!(pospos,d.pt1_cam1_X...)
    end
    ##
    stimdf = DataFrame("pre"=>prepos,"post"=>pospos)
    stimdf.time = 1/300:1/300:(nrow(stimdf)/camfs)
    ##
    predat = glob("$(moth)_$(tris[1])*.h5",joinpath(datadir,moth))[1]
    
    dat = h5_to_df(predat)
    camstart = findfirst(x -> x > 1,dat.camtrig)
    dtimedac = (mothstarts[moth][1] - camstart) / fs
    startindexcam = findfirst(x-> x>dtimedac,stimdf.time)
    stimpre = stimdf.pre[startindexcam:(startindexcam + 10*camfs -1)]

    posdat = glob("$(moth)_$(tris[2])*.h5",joinpath(datadir,moth))[1]
    dat = h5_to_df(predat)
    camstart = findfirst(x -> x > 1,dat.camtrig)
    dtimedac = (mothstarts[moth][2] - camstart) / fs
    startindexcam = findfirst(x-> x>dtimedac,stimdf.time)
    stimpost = stimdf.post[startindexcam:(startindexcam + 10*camfs -1)]


    ##
    return(float.(stimpre),float.(stimpost))
end
##


##
moths = collect(keys(allmoths))
cold_moths = ["2024_08_01","2024_06_06","2024_06_20","2024_06_24","2024_12_04_2"]
moths = [m for m in moths if !in(m,cold_moths)]
##
for m in moths

    pr, po = get_stims(m,datadir,mothstarts)
    vpr = savitzky_golay(pr,21,2,deriv=1).y
    vpo = savitzky_golay(po,21,2,deriv=1).y

    allmoths[m]["stimpre"] = savitzky_golay(pr,21,2,deriv=0).y
    allmoths[m]["stimpost"] = savitzky_golay(po,21,2,deriv=0).y
    allmoths[m]["velpre"] = vpr
    allmoths[m]["velpost"] = vpo
end

##
@save "fat_moths_set_1.jld" allmoths
##

