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
using Associations
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
@load "fat_moths_set_1.jld" allmoths
fs = 1e4 
##
d = allmoths["2024_11_08"]["data"]
subs = select(d,:trial,:wb,:muscle,:phase,:fx_pc1,:fx_pc2,:fx_pc3)
##
firstspike = @pipe d |> 
    groupby(_,[:wb,:muscle]) |> 
    combine(_,:phase=>minimum=>:phase) |>
    unstack(_,:wb,:muscle,:phase)
##
fxt = select(d,:trial,:wb,:fx_pc1,:fx_pc2,:fx_pc3)
unique!(fxt)
leftjoin!(firstspike,fxt,on=:wb)
##