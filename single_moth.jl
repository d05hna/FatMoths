using CSV
using DataFrames
using Pipe
using Glob
using GLMakie
using HDF5
using Interpolations
using LinearAlgebra
using StatsBase
using Optim
using DSP
using DelimitedFiles
using DataFramesMeta
using FFTW
using AlgebraOfGraphics
using JLD2
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:turquoise,:coral],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##

datadir = "/home/doshna/Documents/PHD/data/fatties"
moth = "2024_10_28"
fs = 1e4
## get data and params

df,params = read_ind(datadir,moth,"hilbert")
df.time_abs = df.time_abs .+ 30

## get the side slipping forces out 
fx_pre,fx_post = get_side_slips(datadir,moth,params)
##
"""
STOP RIGHT HERE!!! LOOK Between Videos and Force Traces and Get the Best 10 Seconds of Tracking for each!!
Then Adjust the Following Variables accordingly
Also if the pre and post trials are not 0 and 2, then change those too!!
"""
## Filter the Fx data and the Muscle Data To only care about the Portions that are being used in the tracking analysis
start_pre = 1
start_post = 125000

pre10 = fx_pre[start_pre:Int(start_pre+1e5-1)]
post10 = fx_post[start_post:Int(start_post+1e5-1)]

tris = [0.,1.]

predf = df[df.trial.==tris[1],:]
predf = predf[predf.time_abs .> start_pre /fs .&& predf.time_abs .< (start_pre-1+1e5)/fs,:]
predf.trial .= "pre"

postdf = df[df.trial.==tris[2],:]
postdf = postdf[postdf.time_abs .> start_post/fs .&& postdf.time_abs .< (start_post-1+1e5)/fs,:]
postdf.trial .= "post"

df_to_use = vcat(predf,postdf,cols=:union)

## peak at the wb freq


plot = data(df_to_use)*mapping(:wbfreq,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) |> draw
## Take a look at mean z forces

plot = data(df_to_use)*mapping(:fz,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) |> draw
## save away 
@load "fat_moths_set_1.jld" allmoths

d = Dict(
    "data" => df_to_use,
    "fxpre" => pre10,
    "fxpost" => post10
)

allmoths[moth] = d 

@save "fat_moths_set_1.jld" allmoths

