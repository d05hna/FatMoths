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
moth = "2024_07_09"
fs = 1e4
##

df,_ = read_ind(datadir,moth,"hilbert")
##
df.trial = string.(df.trial)
plot = data(df)*mapping(:tz_pc1,:tz_pc2,row=:trial)*visual(color=:coral) |> draw


##
all_moths= DataFrame()
##
df.group .= "Cold"

all_moths = vcat(all_moths,df)