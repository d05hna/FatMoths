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
# theme.palette = (color = [:turquoise,:coral],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
date = "2024_11_25"
path = "/home/doshna/Documents/PHD/data/fatties/$date/$(date)_003.h5"

df = h5_to_df(path)

fig = Figure(resolution=(1200,300))
ax = Axis(fig[1,1])
lines!(df.time,df.fx)


display(fig)