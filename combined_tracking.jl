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
theme.palette = (; color = [:firebrick,:steelblue])
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
## sub out moths that dont increase the fx+tz + roll more than 10918150% on average 
mc = get_mean_changes(allmoths;axis=6)
moths = mc[mc.mean_gain .> 25,:moth]
n_moths = length(moths)
##
Yawi = zeros(18,n_moths) |> x -> Complex.(x)
Yawf = zeros(18,n_moths) |> x -> Complex.(x)
Fxi = zeros(18,n_moths) |> x -> Complex.(x)
Fxf = zeros(18,n_moths) |> x -> Complex.(x)
Ci = zeros(18,n_moths) |> x -> Complex.(x)
Cf = zeros(18,n_moths) |> x -> Complex.(x)
##
for (i,m) in enumerate(moths)
    skinny = allmoths[m]["ftpre"]
    fat = allmoths[m]["ftpos"]

    sfx = zscore(skinny[:,1])
    syaw = zscore(skinny[:,6])
    
    ffx = zscore(fat[:,1])
    fyaw = zscore(fat[:,6])
    
    ci = sfx .+ -1 .* syaw 
    cf = ffx .+ -1 .* fyaw 
 
    stimpre = zscore(allmoths[m]["stimpre"])

    stimpost = zscore(allmoths[m]["stimpost"])

    Yawi[:,i] = tf_freq(stimpre,syaw,freqqs,fs)
    Yawf[:,i] = tf_freq(stimpost,fyaw,freqqs,fs)
    Fxi[:,i] = tf_freq(stimpre,sfx,freqqs,fs)
    Fxf[:,i] = tf_freq(stimpost,ffx,freqqs,fs)
    Ci[:,i] = tf_freq(stimpre,ci,freqqs,fs)
    Cf[:,i] = tf_freq(stimpost,cf,freqqs,fs)
end
##


##

logx = log10.(freqqs[1:18])
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
ws = right .- left

f = Figure(size=(1100,500)) 
ax = Axis(f[1,1],xscale=log10,xlabel="Freq",ylabel = "Gain",xticks=[0.1,1,10],title="Yaw",yscale=log10)
ax.yticks = [0.1,1]
ax.limits = (0.1,nothing,nothing,nothing)
y = vec(abs.(Yawi))
x = repeat(freqqs,outer = n_moths)
boxplot!(ax,x,y,label="Low Mass",color=(:steelblue,0.5),width=repeat(ws,outer=n_moths))
boxplot!(ax,x,vec(abs.(Yawf)),width = repeat(ws,outer=n_moths),label="High Mass",color=(:firebrick,0.5))
# text!(ax,0.3,1,text = "$(meanyawchange)% Increase",fontsize=14,font=:bold)


ax = Axis(f[2,1],xscale=log10,xlabel="Freq",ylabel = "Gain",xticks=[0.1,1,10],title="Fx",yscale=log10)
ax.yticks=[0.01,0.1,1]
ax.limits = (0.1,nothing,nothing,nothing)

boxplot!(ax,x,vec(abs.(Fxi)),label="Low Mass",color=(:steelblue,0.5),width=repeat(ws,outer=n_moths))
boxplot!(ax,x,vec(abs.(Fxf)),width = repeat(ws,outer=n_moths),label="High Mass",color=(:firebrick,0.5),)
# text!(ax,0.3,0.5,text = "$(meanfxchange)% Increase",fontsize=14,font=:bold)

ax = Axis(f[:,2],xscale=log10,xlabel="Freq",ylabel = "Gain",xticks=[0.1,1,10],title="Combined",yscale=log10)
ax.yticks = [0.1,1]
ax.limits = (0.1,nothing,nothing,nothing)
# text!(ax,0.3,1,text = "$(meancombochange)% Increase",fontsize=14,font=:bold)

boxplot!(ax,x,vec(abs.(Ci)),label="Low Mass",color=(:steelblue,0.5),width=repeat(ws,outer=n_moths))
boxplot!(ax,x,vec(abs.(Cf)),width = repeat(ws,outer=n_moths),label="High Mass",color=(:firebrick,0.5))

f
##

myi = mean(abs.(Yawi),dims=2)[:] 
syi = std(abs.(Yawi),dims=2)[:] ./ sqrt(n_moths)
myf = mean(abs.(Yawf),dims=2)[:] 
syf = std(abs.(Yawf),dims=2)[:] ./ sqrt(n_moths)

myip = unwrap_negative(mean(angle.(Yawi),dims=2)[:])
syip = std(angle.(Yawi),dims=2)[:] ./ sqrt(n_moths)
myfp = unwrap_negative(mean(angle.(Yawf),dims=2)[:])
syfp = std(angle.(Yawf),dims=2)[:] ./ sqrt(n_moths)

mfi = mean(abs.(Fxi),dims=2)[:] 
sfi = std(abs.(Fxi),dims=2)[:] ./ sqrt(n_moths)
mff = mean(abs.(Fxf),dims=2)[:] 
sff = std(abs.(Fxf),dims=2)[:] ./ sqrt(n_moths)

mfip = unwrap_negative(mean(angle.(Fxi),dims=2)[:])
sfip = std(angle.(Fxi),dims=2)[:] ./ sqrt(n_moths)
mffp = unwrap_negative(mean(angle.(Fxf),dims=2)[:])
sffp = std(angle.(Fxf),dims=2)[:] ./ sqrt(n_moths)

mci = mean(abs.(Ci),dims=2)[:] 
sci = std(abs.(Ci),dims=2)[:] ./ sqrt(n_moths)
mcf = mean(abs.(Cf),dims=2)[:] 
scf = std(abs.(Cf),dims=2)[:] ./ sqrt(n_moths)

mcip = unwrap(mean(angle.(Ci),dims=2)[:])
scip = std(angle.(Ci),dims=2)[:] ./ sqrt(n_moths)
mcfp = unwrap(mean(angle.(Cf),dims=2)[:])
scfp = std(angle.(Cf),dims=2)[:] ./ sqrt(n_moths)

f = Figure(size=(1200,800)) 
ax1 = Axis(f[1,1],xscale=log10,yscale=log10,title="Yaw Gain",yticks=[0.01,0.1,1],xticklabelsvisible=false,ylabel="Gain")
ax1.limits = (0.1,nothing,0.01,1.6)
lines!(ax1,freqqs,myi,color=:steelblue,linewidth=3)
lines!(ax1,freqqs,myf,color=:firebrick,linewidth=3)
band!(ax1,freqqs,myi .+ syi,myi .- syi,color=:steelblue,alpha=0.4)
band!(ax1,freqqs,myf .+ syf,myf .- syf,color=:firebrick,alpha=0.4)
ax2 = Axis(f[2,1],xscale=log10,title="Phase",xticks=[0.2,1,10],xlabel="Frequency (Hz)",ylabel="Phase (rad)",yticks=([pi/2,0,-pi/2],[L"\frac{\pi}{2}",L"0",L"-\frac{\pi}{2}"]))
lines!(ax2,freqqs,myip,color=:steelblue,linewidth=3)
lines!(ax2,freqqs,myfp,color=:firebrick,linewidth=3)
band!(ax2,freqqs,myip .+ syip,myip .- syip,color=:steelblue,alpha=0.4)
band!(ax2,freqqs,myfp .+ syfp,myfp .- syfp,color=:firebrick,alpha=0.4)

linkxaxes!(ax2,ax1)

ax3 = Axis(f[1,2],xscale=log10,yscale=log10,title="Fx Gain",yticks=[0.01,0.1,1],xticklabelsvisible=false,ylabel="Gain")
ax3.limits = (0.1,nothing,0.01,1.6)
lines!(ax3,freqqs,mfi,color=:steelblue,linewidth=3)
lines!(ax3,freqqs,mff,color=:firebrick,linewidth=3)
band!(ax3,freqqs,mfi .+ sfi,mfi .- sfi,color=:steelblue,alpha=0.4)
band!(ax3,freqqs,mff .+ sff,mff .- sff,color=:firebrick,alpha=0.4)
ax4 = Axis(f[2,2],xscale=log10,title="Phase",xticks=[0.2,1,10],xlabel="Frequency (Hz)",ylabel="Phase (rad)",yticks=([pi/2,0,-pi/2],[L"\frac{\pi}{2}",L"0",L"-\frac{\pi}{2}"]))
lines!(ax4,freqqs,mfip,color=:steelblue,linewidth=3)
lines!(ax4,freqqs,mffp,color=:firebrick,linewidth=3)
band!(ax4,freqqs,mfip .+ sfip,mfip .- sfip,color=:steelblue,alpha=0.4)
band!(ax4,freqqs,mffp .+ sffp,mffp .- sffp,color=:firebrick,alpha=0.4)

linkxaxes!(ax3,ax4)

ax5 = Axis(f[1,3],xscale=log10,yscale=log10,title="Combined Gain",yticks=[0.01,0.1,1],xticklabelsvisible=false,ylabel="Gain")
ax5.limits = (0.1,nothing,0.01,1.6)
lines!(ax5,freqqs,mci,color=:steelblue,linewidth=3)
lines!(ax5,freqqs,mcf,color=:firebrick,linewidth=3)
band!(ax5,freqqs,mci .+ sci,mci .- sci,color=:steelblue,alpha=0.4)
band!(ax5,freqqs,mcf .+ scf,mcf .- scf,color=:firebrick,alpha=0.4)
ax6 = Axis(f[2,3],xscale=log10,title="Phase",xticks=[0.2,1,10],xlabel="Frequency (Hz)",ylabel="Phase (rad)",yticks=([0,-pi,-2pi],[L"0",L"-\pi",L"-2\pi"]))
lines!(ax6,freqqs,mcip,color=:steelblue,linewidth=3)
lines!(ax6,freqqs,mcfp,color=:firebrick,linewidth=3)
band!(ax6,freqqs,mcip .+ scip,mcip .- scip,color=:steelblue,alpha=0.4)
band!(ax6,freqqs,mcfp .+ scfp,mcfp .- scfp,color=:firebrick,alpha=0.4)

linkxaxes!(ax3,ax4)

# ax2.limits=(0.1,nothing,nothing,nothing)
# save("Figs/GoodFigs/SubSetTracking.png",f,px_per_unit=4)
f