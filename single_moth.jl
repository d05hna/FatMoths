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
using MultivariateStats
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
global ftnames = ["fx","fy","fz","tx","ty","tz"]
datadir = "/home/doshna/Documents/PHD/data/fatties"
moth = "2024_11_05"
fs = 1e4
## get data and params

df,params = read_ind(datadir,moth,"hilbert")
df.time_abs = df.time_abs .+ 30

## get the side slipping forces out 
fx_pre,fx_post = get_side_slips(datadir,moth,params,[1,4])
##
"""
STOP RIGHT HERE!!! LOOK Between Videos and Force Traces and Get the Best 10 Seconds of Tracking for each!!
Then Adjust the Following Variables accordingly
Also if the pre and post trials are not 0 and 2, then change those too!!
"""
## Filter the Fx data and the Muscle Data To only care about the Portions that are being used in the tracking analysis
start_pre = Int(1)
start_post = Int(1e5)

pre10 = fx_pre[start_pre:Int(start_pre+1e5-1)]
post10 = fx_post[start_post:Int(start_post+1e5-1)]

tris = [0.,3.]
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
## Take a look at Muscle Activities
plot = data(df_to_use) * mapping(:phase,row=:muscle,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=250,normalization=:probability)*visual(alpha=0.5) |> draw
## Take a Look at the frequency Responses 
fftpre = abs.(fft(pre10)[2:50000])
fftpost = abs.(fft(post10)[2:50000])
freqrange = fftfreq(length(pre10),fs)[2:50000]
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

id = findfirst(x->x ==15, freqrange)
fftpre = fftpre[1:id]
fftpost=fftpost[1:id]
freqrange=freqrange[1:id]

fig = Figure()
ax1 = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Pre",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)
    )
ax2 = Axis(fig[1, 2], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Post",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)
    )
ax3 = Axis(fig[2, :], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Both",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)

    )


pre = lines!(ax1, freqrange,fftpre,color=:turquoise)
post = lines!(ax2, freqrange,fftpost,color=:coral)

lines!(ax3, freqrange,fftpre,color=:turquoise)
lines!(ax3, freqrange,fftpost,color=:coral)

vlines!(ax1,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
vlines!(ax2,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
vlines!(ax3,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

Legend(fig[:,:3],[pre,post],["Before Feeding","After Feeding"])
fig
## save away
@load "fat_moths_set_1.jld" allmoths

d = Dict(
    "data" => df_to_use,
    "fxpre" => pre10,
    "fxpost" => post10
)

allmoths[moth] = d 

@save "fat_moths_set_1.jld" allmoths

