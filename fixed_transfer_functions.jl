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
using Colors
using GLM
using SavitzkyGolay
using HypothesisTests

include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_black()
theme.palette = (color = [:steelblue,:firebrick],)
theme.fontsize = 20
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
@load "fat_moths_set_1.jld" allmoths
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
##
moth = collect(keys(allmoths))[1]
for moth in collect(keys(allmoths))
    pr,po = transfer_function(allmoths,moth;axis="tz")

    unpre = unwrap(angle.(pr))
    unpo = unwrap(angle.(po))

    f = Figure()
    ax = Axis(f[1,1],xscale=log10,xlabel="freq (hz)",ylabel="gain")
    scatter!(ax,freqqs,abs.(pr),color=:steelblue)
    scatter!(ax,freqqs,abs.(po),color=:firebrick)
    ax2 = Axis(f[2,1],xscale=log10,xlabel="freq (hz)",ylabel="phase")
    scatter!(ax2,freqqs,unpre,color=:steelblue)
    scatter!(ax2,freqqs,unpo,color=:firebrick)
    ax.title=moth
    save("fixed_tf/$moth.png",f)
end