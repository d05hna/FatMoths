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
using MultivariateStats
using Distributions
# using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)

@load "fat_moths_set_1.jld" allmoths
pixel_conversion = 0.14 ## mm/pixels 


freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
fs = Int(1e4)
N = Int(1e5)
pixel_conversion = 0.14 ## mm/pixels 

delete!(allmoths,"2025_10_22")
delete!(allmoths,"2025_10_21")
##
fr = round.(fftfreq(N,fs),digits=4)
stimidx= [i for i in 1:N if fr[i] in freqqs]
c_pre = zeros(18,16)
c_post = zeros(18,16)
f_pre = zeros(18,16)
f_post = zeros(18,16)
y_pre = zeros(18,16)
y_post = zeros(18,16)
for (i,m) in enumerate(collect(keys(allmoths)))
    skinny = allmoths[m]["ftpre"]
    fat = allmoths[m]["ftpos"]

    sfx = zscore(skinny[:,1])
    syaw = zscore(skinny[:,6])

    ffx = zscore(fat[:,1])
    fyaw = zscore(fat[:,6])

    ci = sfx .+ 1 .* syaw 
    cf = ffx .+ 1 .* fyaw 

    stimpre = allmoths[m]["stimpre"] 
    itp = interpolate(stimpre, BSpline(Linear()))
    stimpre = itp(LinRange(1, length(itp), Int(1e5)))
    stimpost = allmoths[m]["stimpost"] 
    itp = interpolate(stimpost, BSpline(Linear()))
    stimpost = itp(LinRange(1, length(itp), Int(1e5)))

    mpre = mt_coherence(hcat(ci,zscore(stimpre))';fs=fs,nfft=N)
    mpo = mt_coherence(hcat(cf,zscore(stimpost))';fs=fs,nfft=N)
    
    c_pre[:,i] = mpre.coherence[1,2,stimidx]
    c_post[:,i] = mpo.coherence[1,2,stimidx]

    mpre = mt_coherence(hcat(syaw,zscore(stimpre))';fs=fs,nfft=N)
    mpo = mt_coherence(hcat(fyaw,zscore(stimpost))';fs=fs,nfft=N)
    
    y_pre[:,i] = mpre.coherence[1,2,stimidx]
    y_post[:,i] = mpo.coherence[1,2,stimidx]

    mpre = mt_coherence(hcat(sfx,zscore(stimpre))';fs=fs,nfft=N)
    mpo = mt_coherence(hcat(ffx,zscore(stimpost))';fs=fs,nfft=N)
    
    f_pre[:,i] = mpre.coherence[1,2,stimidx]
    f_post[:,i] = mpo.coherence[1,2,stimidx]
end
##


fre = repeat(1:18,outer=16)
mo = repeat(collect(keys(allmoths)),inner=18)

bigdf = DataFrame()
dfpre = DataFrame(mo = mo, freq = fre,coh = vec(c_pre))
dfpre.trial .= "Low Mass" 
dfpost = DataFrame(mo=mo,freq=fre,coh=vec(c_post))
dfpost.trial .= "High Mass"
df = vcat(dfpre,dfpost)
df.axis .= "combined"
bigdf = vcat(bigdf,df,cols=:union)

dfpre = DataFrame(mo = mo, freq = fre,coh = vec(y_pre))
dfpre.trial .= "Low Mass" 
dfpost = DataFrame(mo=mo,freq=fre,coh=vec(y_post))
dfpost.trial .= "High Mass"
df = vcat(dfpre,dfpost)
df.axis .= "yaw"
bigdf = vcat(bigdf,df,cols=:union)

dfpre = DataFrame(mo = mo, freq = fre,coh = vec(f_pre))
dfpre.trial .= "Low Mass" 
dfpost = DataFrame(mo=mo,freq=fre,coh=vec(f_post))
dfpost.trial .= "High Mass"
df = vcat(dfpre,dfpost)
df.axis .= "fx"
bigdf = vcat(bigdf,df,cols=:union)

plot = data(bigdf)*
    mapping(:freq => "Frequency (Hz)",:coh => "Coherence",row=:trial => renamer("Low Mass" => "Low Mass","High Mass"=>"High Mass"),color=:axis,dodge=:axis)*
    visual(BoxPlot)
f = draw(plot,figure = (; size=(1200,600)),axis=(; xticks=(1:18,string.(freqqs))))
save("Figs/GoodFigs/AllCoherence.png",f,px_per_unit=4)
f
##
""" 
Coherence is much Better in Yaw Especially As Freqqs Get higher 
""" 
