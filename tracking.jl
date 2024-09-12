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
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
##
datadir = "/home/doshna/Documents/PHD/data/fatties/"
moth = "2024_08_16"

files = glob("$(moth)/*.h5",datadir)

##
fs = 10000
ftnames = ["fx","fy","fz","tx","ty","tz"]

df = h5_to_df(files[1])

true_ft = DataFrame(transform_FT(transpose(Matrix(df[!,ftnames]))),ftnames)

df[!,ftnames] = true_ft
idx = findall(df.camtrig .> 2)[1]

fx = df.fx[idx:idx+ fs*20-1]
##

stims = glob("*/*DLT*/*",datadir)
s = stims[4]
stim = CSV.read(s,DataFrame)

stim_20 = convert(Vector{Float64},stim.filtered[1:6000])
##
fftstim_pre = fft(stim_20)[2:3000]
freq_range_pre = fftfreq(length(stim_20),300)[2:3000]
mag_stim_pre = abs.(fftstim_pre)

freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70,16.70 ,19.90]

fftfx_pre = fft(fx)[2:100000]
fr_fx_pre = fftfreq(length(fx),10000)[2:100000]
mag_fx_pre = abs.(fftfx_pre)
##
dfpost = h5_to_df(files[4])

true_ftpost = DataFrame(transform_FT(transpose(Matrix(df[!,ftnames]))),ftnames)

df[!,ftnames] = true_ftpost
idx = findall(df.camtrig .> 2)[1]

fxpost = df.fx[idx:idx+ fs*20-1]
##
s = stims[end]
stimpost = CSV.read(s,DataFrame)

stim_20post = convert(Vector{Float64},stim.filtered[1:6000])
##
fftstimpost = fft(stim_20post)[2:3000]
freq_range_post = fftfreq(length(stim_20),300)[2:3000]
mag_stim_post = abs.(fftstimpost)

freqqs_post = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70,16.70 ,19.90]

fftfx_post = fft(fxpost)[2:100000]
fr_fx_post = fftfreq(length(fxpost),10000)[2:100000]
mag_fx_post = abs.(fftfx_post)

##
f = Figure(figsize=(6,6))
ax = Axis(f[1,1],xscale=log10)
lines!(ax,freq_range_pre,mag_stim_pre)
fr = vlines!(ax,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

ax2 = Axis(f[2,1],xscale=log10)
lines!(ax2,fr_fx_pre,mag_fx_pre)
fr = vlines!(ax2,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

ax3 = Axis(f[1,2],xscale=log10)
lines!(ax3,freq_range_post,mag_stim_post)
fr = vlines!(ax3,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

ax4 = Axis(f[2,2],xscale=log10)
lines!(ax4,fr_fx_post,mag_fx_post)
fr = vlines!(ax4,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

ax.title = "Flower Pre"
ax2.title = "Moth Pre"
ax3.title = "Flower Post"
ax4.title = "Moth Post"

save("Moth_08_16_24/080124Mothfft.png",f,px_per_unit=4)
f
##

idxs = findall(x->x in freqqs,round.(fr_fx_pre,digits=2))

gains_pre = [log10(mag_fx_pre[i]/mag_stim_pre[i]) for i in idxs]
gains_post = [log10(mag_fx_post[i]/mag_stim_post[i]) for i in idxs]

g = Figure()
ax = Axis(g[1,1],xscale=log10)

pre = scatter!(ax,freqqs,gains_pre)
post = scatter!(ax,freqqs,gains_post)

Legend(g[1,2],[pre,post],["Pre","Post"])
ax.title="Sensorimotor Gain"
ax.ylabel = "Log10 newtons/pixels?"
ax.xlabel = "Frequency"
save("Moth_08_16_24/gains.png",g,px_per_unit=4)
g