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
Hi = zeros(18,10) |> x -> Complex.(x)
Hf = zeros(18,10) |> x -> Complex.(x)
N = 2000
fr = round.(fftfreq(N,100),digits=4)
stimidx= [i for i in 1:N if fr[i] in freqqs]


for (i,m) in enumerate(collect(keys(FreeFlight)))
    xi = FreeFlight[m]["moth_pre"]
    xf = FreeFlight[m]["moth_post"]
    fi = FreeFlight[m]["flower_pre"]
    ff = FreeFlight[m]["flower_post"]

    hi = fft(xi) ./ fft(fi)
    hf = fft(xf) ./ fft(ff)

    Hi[:,i] = hi[stimidx]
    Hf[:,i] = hf[stimidx]
end
##
pre = DataFrame() 
post = DataFrame() 

@showprogress for i in 1:18 
    realimagi = hcat(real.(Hi[i,:]),imag.(Hi[i,:]))
    coi = cov(realimagi)
    meandi = mean(realimagi,dims=1)[:]

    realimagf = hcat(real.(Hf[i,:]),imag.(Hf[i,:]))
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
CSV.write("Steven/CL_pre_stats.csv",pre)
CSV.write("Steven/CL_post_stats.csv",post)
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
save("Figs/PaperFigs/CL_tracking.svg",F,px_per_unit=4)
F
##
""" 
Tracking Error Time 
""" 

rpre = real.(Hi)
rpost = real.(Hf)
ipre = imag.(Hi)
ipost = imag.(Hf)

drpre = (rpre .- 1).^2
drpost = (rpost .-1).^2

dipre = (ipre .- 0).^2
dipost = (ipost .-0) .^2

epre = sqrt.(drpre .+ dipre)
epost = sqrt.(drpost .+ dipost)

meanepre = mean(epre,dims=2)[:]
meanpost = mean(epost,dims=2)[:]

ci_pre = [ci_low_n(epre[i,:]) for i in 1:18]
ci_post = [ci_low_n(epost[i,:]) for i in 1:18]

F = Figure(size=(800,400))
ax = Axis(F[1,1],xlabel="Frequency",xscale=log10,xticks=[0.1,1,10],
    ylabel = "Tracking Error",yticks=[0,1,2],limits=(0.1,nothing,nothing,nothing))
lines!(ax,freqqs,meanepre,linewidth=3,color=:steelblue)
errorbars!(ax,freqqs,meanepre,ci_pre,whiskerwidth=10,color=:steelblue)
lines!(ax,freqqs,meanpost,linewidth=3,color=:firebrick)
errorbars!(ax,freqqs,meanpost,ci_post,whiskerwidth=10,color=:firebrick)
save("Figs/PaperFigs/Tracking_error.svg",F,px_per_unit=4)
F