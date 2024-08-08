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
using CategoricalArrays
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("/home/doshna/Documents/PHD/FatMoths/me_functions.jl")
##

##

datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")
moth = "2024_08_01"
##
df = read_ind_ldlm(datadir,moth)

## lets look at wbfreq 

d4t = @view df[!,:]
trial_to_label = Dict(0 => "pre",1=>"mid",2=>"post")
d4t[!, :trial] = map(i -> trial_to_label[i], d4t.trial)
##
plot = data(d4t)*mapping(:wbfreq,color=:trial)*histogram(bins=50,normalization=:pdf)*visual(alpha=0.7) 
fig = draw(plot,axis=(;))
save("Moth_08_01_24/freq_with_ldlm.png",fig,px_per_unit=3)
fig
## lets look at muscle phase

plot=data(d4t)*mapping(:phase,layout=:muscle,color=:trial)*AlgebraOfGraphics.histogram(bins=100)*visual(alpha=0.5)
fig = draw(plot,axis=(;))
save("Moth_08_01P_24/musc_with_ldlm.png",fig,px_per_unit=3)
fig

