using CSV
using DataFrames
using Pipe
using Glob
using CairoMakie
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
using MultivariateStats
using Distributions
using QuadGK
using ProgressMeter
using MAT
# using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/functions.jl")
include("me_functions.jl")
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moth_free_flight.jld" FreeFlight
pixel_conversion = 0.14 ## mm/pixels 


freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
##

pnt = matread("Steven/steven_data_linear.mat")["results"]
Ci = hcat(pnt["Ci"]...)
Cf = hcat(pnt["Cf"]...)



##
pre = DataFrame() 
post = DataFrame() 

@showprogress for i in 1:18 
    realimagi = hcat(real.(Ci[i,:]),imag.(Ci[i,:]))
    coi = cov(realimagi)
    meandi = mean(realimagi,dims=1)[:]

    realimagf = hcat(real.(Cf[i,:]),imag.(Cf[i,:]))
    cof = cov(realimagf)
    meandf = mean(realimagf,dims=1)[:]


    gi,ti = confMagPhase(meandi,coi,0.95)
    mg = norm(meandi)
    mp = atan(meandi[2],meandi[1])
    tmp = Dict("freq"=> freqqs[i],"mg" => mg,"mp"=>mp,"glow"=>mg - gi[1],"ghigh" => gi[2]-mg,"plow"=>mp - ti[1],"phigh"=>ti[2]-mp)
    push!(pre,tmp,cols=:union)

    gf,tf = confMagPhase(meandf,cof,0.95)
    mg = norm(meandf)
    mp = atan(meandf[2],meandf[1])
    tmp = Dict("freq"=> freqqs[i],"mg" => mg,"mp"=>mp,"glow"=>mg- gf[1],"ghigh" => gf[2]-mg,"plow"=>mp-tf[1],"phigh"=>tf[2]-mp)
    push!(post,tmp,cols=:union)
end
pre.mp = unwrap_negative(pre.mp)
post.mp = unwrap_negative(post.mp)

##
F = Figure(size=(800,800)) 
ax = Axis(F[2,1],ylabel="Gain",xscale=log10,xticklabelsvisible=false,yticks=[0,0.1,1],yscale=log10)

lines!(ax,pre.freq,pre.mg,color=:steelblue,linewidth=3)
lines!(ax,post.freq,post.mg,color=:firebrick,linewidth=3)
errorbars!(ax,pre.freq,pre.mg,pre.glow,pre.ghigh,color=:steelblue,whiskerwidth=10)
errorbars!(ax,post.freq,post.mg,post.glow,post.ghigh,color=:firebrick,whiskerwidth=10)
ax2 = Axis(F[3,1],ylabel="Phase (radians)",xlabel="Frequency (Hz)",xscale=log10,xticks=[0.1,1,10],limits=(0.1,nothing,nothing,nothing))

l = lines!(ax2,pre.freq,pre.mp,color=:steelblue,linewidth=3)
h = lines!(ax2,post.freq,post.mp,color=:firebrick,linewidth=3)
errorbars!(ax2,pre.freq,pre.mp,pre.plow,pre.phigh,whiskerwidth=10,color=:steelblue)
errorbars!(ax2,post.freq,post.mp,post.plow,post.phigh,whiskerwidth=10,color=:firebrick)

linkxaxes!(ax2,ax)
Legend(F[1,1],[l,h],["Low Mass","High Mass"],orientation = :horizontal)
save("Figs/PaperFigs/OL_tracking.svg",F,px_per_unit=4)
F
##
gci = abs.(Ci)
gcf = abs.(Cf)

gchange = gcf ./ gci
mgchange = mean(gchange,dims=2)[:]
semchange = std(gchange,dims=2)[:] ./sqrt(10)

logx = log10.(freqqs)
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
widths = right .- left

f = Figure(size=(800,400)) 
ax = Axis(f[1,1],xscale=log10,limits=(0.15,18,0,4))
ax.xticks=[0.2,1,10]
ax.xlabel="Frequency (Hz)"
ax.ylabel = "Gain Change Multiple"
barplot!(ax,freqqs,mgchange,width=widths,color=:grey)
errorbars!(ax,freqqs,mgchange,semchange,color=:black)
lines!(ax, range(0.15,18,length=16),repeat([1.7],16),linestyle=:dash,color=:red)
save("Figs/PaperFigs/barplot_OL.svg",f,px_per_unit=4)
f