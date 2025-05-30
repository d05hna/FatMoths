using CSV 
using DataFrames
using HDF5
using JLD2
using SavitzkyGolay
using Pipe
using DataFramesMeta
using Glob
include("me_functions.jl")
##

gtp = Dict(
    ## Moth => Trial 1, Trial 2, Start 1, Start 2, Trials are 1 indexed
    "2024_11_01" => [2,4,1.5e5,1],
    "2024_11_04" => [2,4,1.5e5,1],
    "2024_11_05" => [1,4,1,1e5],
    "2024_11_07" => [1,3,1e4,1],
    "2024_11_08" => [2,4,1e5,1],
    "2024_11_11" => [1,2,1,1],
    "2024_11_20" => [1,3,1e5,1.5e5],
    "2025_01_30" => [1,2,1e5,1.5e5],
    "2025_03_20" => [1,2,5e4,5e4],
    "2025_04_02" => [1,2,5e4,1e5],


)
##
@load "fat_moths_set_1.jld" allmoths
datadir = "/home/doshna/Documents/PHD/data/fatties/"
stimdic = h5_to_dict(joinpath(datadir,"DLC/raw_mats.h5"))

##
dac_fs = Int(1e4)
cam_fs = Int(300)
for moth in collect(keys(allmoths))
    tris = ["00$(Int(gtp[moth][1]-1))","00$(Int(gtp[moth][2]-1))"]
    for (i,tri) in enumerate(["pre","post"])
        t = tris[i] 
        nam = sort(filter(x->occursin("$(replace(moth,"_"=>""))_$(t)",x), collect(keys(stimdic))))
        alltris = DataFrame()
        for n in nam
            tmp = DataFrame(stimdic[n], stimdic["names"])
            tmp.part.= n
            alltris = vcat(alltris,tmp,cols=:union)
        end
        alltris.time = range(start=1/300,step=1/300,stop=nrow(alltris)/300)
        datpath = glob("$(moth)_$(tris[i])*.h5",joinpath(datadir,moth))[1]
        println(datpath)
        dat = h5_to_df(datpath)
        camstart = findfirst(x -> x > 1,dat.camtrig)
        start_time = gtp[moth][i+2]
        dtimedac = (start_time - camstart) / dac_fs

        startindexcam = findfirst(x-> x>dtimedac,alltris.time)
        flower_pos = alltris.x[startindexcam:(startindexcam + 10*cam_fs -1)]
        flower_pos_smooth = savitzky_golay(flower_pos,21,2,deriv=0).y
        flower_vel_smooth = savitzky_golay(flower_pos,21,2,deriv=1).y
        allmoths[moth]["stim$(tri)"] = flower_pos_smooth
        allmoths[moth]["vel$(tri)"] = flower_vel_smooth
    end
end
@save "fat_moths_set_1.jld" allmoths
##
