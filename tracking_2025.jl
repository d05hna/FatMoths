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
only_fz = true
##
mc = get_mean_changes(allmoths)
changed_moths = mc[mc.mean_fz .> 0.05,:moth]

if only_fz 
    moths = changed_moths 
    n_moths = length(changed_moths)
else 
    moths = mc.moth
    n_moths = length(moths)
end



Fxi = zeros(ComplexF64,length(freqqs),n_moths)
Fxf = zeros(ComplexF64,length(freqqs),n_moths)
yawi = zeros(ComplexF64,length(freqqs),n_moths)
yawf = zeros(ComplexF64,length(freqqs),n_moths)

for (i,m) in enumerate(moths)
    stimpre = allmoths[m]["stimpre"]
    stimpost = allmoths[m]["stimpost"]
    fxpre = allmoths[m]["ftpre"][:,1]
    fxpost = allmoths[m]["ftpos"][:,1]
    yawpre = allmoths[m]["ftpre"][:,6]
    yawpost = allmoths[m]["ftpos"][:,6]

    Fxi[:,i] = tf_freq(stimpre,fxpre,freqqs,fs)
    Fxf[:,i] = tf_freq(stimpost,fxpost,freqqs,fs)
    yawi[:,i] = tf_freq(stimpre,yawpre,freqqs,fs)
    yawf[:,i] = tf_freq(stimpost,yawpost,freqqs,fs)
end
## Get the stats
XI = DataFrame()
XF = DataFrame()
YI = DataFrame()
YF = DataFrame()
for i in 1:16
    hi = Fxi[i,:]
    hf = Fxf[i,:]
    ci = yawi[i,:]
    cf = yawf[i,:]

    m,r,t = mean_and_ci(hi)
    te,tel,teh = mean_ci_tdist(tracking_error.(hi))
    tmp = Dict(
        "freq" => freqqs[i],
        "mean_gain" => abs.(m),
        "mean_phase" => angle.(m),
        "g_lo" => r[1],
        "g_hi" => r[2],
        "p_lo" => t[1],
        "p_hi" => t[2],
        "mean_error" => te,
        "low_error" => tel,
        "high_error" => teh
    )
    push!(XI,tmp,cols=:union)
    m,r,t = mean_and_ci(hf)
    te,tel,teh = mean_ci_tdist(tracking_error.(hf))
    tmp = Dict(
        "freq" => freqqs[i],
        "mean_gain" => abs.(m),
        "mean_phase" => angle.(m),
        "g_lo" => r[1],
        "g_hi" => r[2],
        "p_lo" => t[1],
        "p_hi" => t[2],
        "mean_error" => te,
        "low_error" => tel,
        "high_error" => teh
    )
    push!(XF,tmp,cols=:union)
    m,r,t = mean_and_ci(ci)
    te,tel,teh = mean_ci_tdist(tracking_error.(ci))
    tmp = Dict(
        "freq" => freqqs[i],
        "mean_gain" => abs.(m),
        "mean_phase" => angle.(m),
        "g_lo" => r[1],
        "g_hi" => r[2],
        "p_lo" => t[1],
        "p_hi" => t[2],
        "mean_error" => te,
        "low_error" => tel,
        "high_error" => teh
    )
    push!(YI,tmp,cols=:union)
    m,r,t = mean_and_ci(cf)
    te,tel,teh = mean_ci_tdist(tracking_error.(cf))
    tmp = Dict(
        "freq" => freqqs[i],
        "mean_gain" => abs.(m),
        "mean_phase" => angle.(m),
        "g_lo" => r[1],
        "g_hi" => r[2],
        "p_lo" => t[1],
        "p_hi" => t[2],
        "mean_error" => te,
        "low_error" => tel,
        "high_error" => teh
    )
    push!(YF,tmp,cols=:union)
end

XI.mean_phase = unwrap_negative(XI.mean_phase)
XF.mean_phase = unwrap_negative(XF.mean_phase)
YI.mean_phase = unwrap_negative(YI.mean_phase)
YF.mean_phase = unwrap_negative(YF.mean_phase)





##

f = Figure(size=(1200,1000)) 
# Fx Gain 
ax = Axis(f[1,1],xscale=log10)
lines!(ax,XI.freq,XI.mean_gain,color=:steelblue,label="Low Mass",linewidth=4)
lines!(ax,XF.freq,XF.mean_gain,color=:firebrick,label="High Mass",linewidth=4)
errorbars!(ax,XI.freq,XI.mean_gain,XI.g_lo,XI.g_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax,XF.freq,XF.mean_gain,XF.g_lo,XF.g_hi,color=:firebrick,whiskerwidth=10)
ax.xticklabelsvisible = false
ax.ylabel = "Gain (mN/ mm)"
ax.title="Fx"

# Yaw Gain
ax2 = Axis(f[2,1],xscale=log10)
lines!(ax2,YI.freq,YI.mean_gain,color=:steelblue,label="Low Mass",linewidth=4)
lines!(ax2,YF.freq,YF.mean_gain,color=:firebrick,label="High Mass",linewidth=4)
errorbars!(ax2,YI.freq,YI.mean_gain,YI.g_lo,YI.g_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax2,YF.freq,YF.mean_gain,YF.g_lo,YF.g_hi,color=:firebrick,whiskerwidth=10)
ax2.xticks = [0.2,0.5,1,5,10]
ax2.ylabel = "Gain (mN-mm / mm)"
ax2.xlabel = "Frequency (Hz)"
ax2.title = "Yaw"
ax2.xlabelfont=:bold

# Fx Phase
ax3 = Axis(f[1,2],xscale=log10)
lines!(ax3,XI.freq,XI.mean_phase,color=:steelblue,label="Low Mass",linewidth=4)
lines!(ax3,XF.freq,XF.mean_phase,color=:firebrick,label="High Mass",linewidth=4)
errorbars!(ax3,XI.freq,XI.mean_phase,XI.p_lo,XI.p_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax3,XF.freq,XF.mean_phase,XF.p_lo,XF.p_hi,color=:firebrick,whiskerwidth=10)
ax3.xticklabelsvisible = false
ax3.ylabel = "Phase (radians)"


ax4 = Axis(f[2,2],xscale=log10)
lines!(ax4,YI.freq,YI.mean_phase,color=:steelblue,label="Low Mass",linewidth=4)
lines!(ax4,YF.freq,YF.mean_phase,color=:firebrick,label="High Mass",linewidth=4)
errorbars!(ax4,YI.freq,YI.mean_phase,YI.p_lo,YI.p_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax4,YF.freq,YF.mean_phase,YF.p_lo,YF.p_hi,color=:firebrick,whiskerwidth=10)
ax4.xticks = [0.2,0.5,1,5,10]
ax4.ylabel = "Phase (radians)"
ax4.xlabel = "Frequency (Hz)"
ax4.xlabelfont=:bold



# Fx Coherence
# ax5 = Axis(f[1,3],xscale=log10)
# lines!(ax5,pre.freq,pre.coh_fx,color=:steelblue,label="Low Mass",linewidth=4)
# lines!(ax5,post.freq,post.coh_fx,color=:firebrick,label="High Mass",linewidth=4)
# band!(ax5,pre.freq,pre.coh_fx - pre.sem_fx_coh,pre.coh_fx + pre.sem_fx_coh,color=:steelblue,alpha=0.2)
# band!(ax5,post.freq,post.coh_fx - post.sem_fx_coh,post.coh_fx + post.sem_fx_coh,color=:firebrick,alpha=0.2)
# ax5.xticklabelsvisible = false
# ax5.ylabel = "Coherence"
# ax5.limits=(nothing,nothing,0,1)

# ax6 = Axis(f[2,3],xscale=log10)
# lines!(ax6,pre.freq,pre.coh_yaw,color=:steelblue,label="Low Mass",linewidth=4)
# lines!(ax6,post.freq,post.coh_yaw,color=:firebrick,label="High Mass",linewidth=4)
# band!(ax6,pre.freq,pre.coh_yaw - pre.sem_yaw_coh,pre.coh_yaw + pre.gain_yaw ./ pre.sem_yaw_coh,color=:steelblue,alpha=0.2)
# band!(ax6,post.freq,post.coh_yaw - post.sem_yaw_coh,post.coh_yaw + post.sem_yaw_coh,color=:firebrick,alpha=0.2)
# ax6.xticks = [0.2,0.5,1,5,10]
# ax6.ylabel = "Coherence"
# ax6.xlabel = "Frequency (Hz)"
# ax6.limits=(nothing,nothing,0,1)
Legend(f[3,1:2],ax,orientation=:horizontal)
save("Figs/NewTrackingFigs/BigTracking_CI.png",f,px_per_unit=4)
f
## gain multiples restricted 

Fxmult = abs.(Fxf) ./ abs.(Fxi)
Yawmult = abs.(yawf) ./ abs.(yawi)
mean_fx = mean(Fxmult,dims=2)[:][1:16]
mean_yaw = mean(Yawmult,dims=2)[:][1:16]
sem_fx = (std(Fxmult,dims=2) ./ sqrt(size(Fxmult,2)))[:][1:16]
sem_yaw = (std(Yawmult,dims=2) ./ sqrt(size(Yawmult,2)))[:][1:16]

mmfx = round(mean(mean_fx),digits=2)
mmyaw = round(mean(mean_yaw),digits=2)

logx = log10.(freqqs[1:16])
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
widths = right .- left

Fig = Figure(size=(1000,600)) 
ax = Axis(Fig[1,1],title="Fx",xticklabelsvisible=false,ylabel="Gain Change Mulitple",xscale=log10,limits=(nothing,nothing,0,4))
barplot!(ax,freqqs[1:16], mean_fx,width=widths,color=:orchid)
lines!(ax,[0.1,10],[1,1],linestyle=:dash, color=:black,linewidth=3)
errorbars!(ax,freqqs[1:16],mean_fx,sem_fx,color=:black,linewidth=3)
lines!(ax,[0.1,10],[mmfx,mmfx],color=:green,label = "Mean Gain Increase = $mmfx")
axislegend(ax)

ax2 = Axis(Fig[2,1],title="Yaw",ylabel="Gain Change Mulitple",xscale=log10,limits=(nothing,nothing,0,4))
barplot!(ax2,freqqs[1:16], mean_yaw,width=widths,color=:orchid)
lines!(ax2,[0.1,10],[1,1],linestyle=:dash, color=:black,linewidth=3)
errorbars!(ax2,freqqs[1:16],mean_yaw,sem_yaw,color=:black,linewidth=3)
lines!(ax2,[0.1,10],[mmyaw,mmyaw],color=:green,label = "Mean Gain Increase = $mmyaw")
axislegend(ax2)
ax2.xticks = [0.1,1,10]
ax2.xlabel = "Frequency (Hz)"
ax2.xlabelfont=:bold
save("Figs/NewTrackingFigs/GainChangeMultiple_madrestricted.png",Fig)
Fig
##
mc = get_mean_changes(allmoths)
mc.masspct = mc.mass_change ./ mc.mass
fatter_moths_list = mc[mc.masspct .> 0.1,:moth]

# fatter_moths = all_data[.!in.(all_data.moth, Ref(fatter_moths_list)), :]

gain_change = combine(groupby(all_data,[:freq])) do gdf 
    prevals = gdf[gdf.condition.=="pre",:]
    postvals = gdf[gdf.condition.=="post",:]
    yawchange = postvals.gain_yaw ./ prevals.gain_yaw
    fxchange = postvals.gain_fx ./ prevals.gain_fx

    (
    yaw_change = mean(yawchange),
    yaw_change_sem = std(yawchange) / sqrt(nrow(gdf)),
    fx_change = mean(fxchange),
    fx_change_sem = std(fxchange) / sqrt(nrow(gdf)),
    yaw_low = mean(yawchange)- (std(yawchange) / sqrt(nrow(gdf))),
    yaw_high = mean(yawchange)+ (std(yawchange) / sqrt(nrow(gdf))),
    fx_low = mean(fxchange) - (std(fxchange) / sqrt(nrow(gdf))),
    fx_high = mean(fxchange) + (std(fxchange) / sqrt(nrow(gdf)))
    )
end

## I ADMIT THIS IS A TERRIBLE WAY TO DO THIS BUT IT WORKS LOLOLOLOLOL
gchange = stack(select(gain_change,:freq,:yaw_change,:fx_change),Not(:freq),variable_name = :axis,value_name=:change)
gchange.axis = [split(x,"_")[1] for x in gchange.axis]
tmp = stack(select(gain_change,:freq,:yaw_change_sem,:fx_change_sem),Not(:freq),variable_name = :axis,value_name=:sem)
tmp.axis = [split(x,"_")[1] for x in tmp.axis]
gchange = leftjoin(gchange,tmp,on=[:freq,:axis])
tmp = stack(select(gain_change,:freq,:yaw_high,:fx_high),Not(:freq),variable_name = :axis,value_name=:high)
tmp.axis = [split(x,"_")[1] for x in tmp.axis]
gchange = leftjoin(gchange,tmp,on=[:freq,:axis])
tmp = stack(select(gain_change,:freq,:yaw_low,:fx_low),Not(:freq),variable_name = :axis,value_name=:low)
tmp.axis = [split(x,"_")[1] for x in tmp.axis]
gchange = leftjoin(gchange,tmp,on=[:freq,:axis])

tmpfreqs = range(1,18,length=18)
axisref = Dict(freqqs[k] => tmpfreqs[k] for k in 1:18)
gchange.tmpf = [axisref[x] for x in gchange.freq]
gchange.null .= 1.0
gchange.yawmean .= mean(gchange[gchange.axis.=="yaw",:change])
gchange.fxmean .= mean(gchange[gchange.axis.=="fx",:change])

##
theme.palette = (; color=([:orchid,:green]))
set_theme!(theme)
plt = data(gchange)*mapping(:tmpf,:change,color=:axis => "Force/Torque Axis",dodge=:axis)*visual(BarPlot)
plt2 = data(gchange) * mapping(:tmpf, :change, :sem, dodge_x = :axis) * visual(Errorbars)
plt3 = data(gchange)*mapping(:tmpf,:null)*visual(Lines,linestyle=:dash,color=:black,linewidth=3,alpha=0.5)
plt4 = data(gchange)*mapping(:tmpf,:yawmean)*visual(Lines,linestyle=:dash,color=:green,linewidth=3,alpha=0.5)
plt5 = data(gchange)*mapping(:tmpf,:fxmean)*visual(Lines,linestyle=:dash,color=:orchid,linewidth=3,alpha=0.5)

fig= draw(plt+plt2+plt3+plt4+plt5,figure=(; size =(1200,400)),axis=(; xticks = ([unique(gchange.tmpf)...],string.(freqqs)),
    xlabel="Frequency (Hz)",xlabelfont=:bold,ylabel="Gain Change Multiple",ylabelfont=:bold))
save("Figs/NewTrackingFigs/GainChangeMultipleBoth.png",fig)
fig
##
gain_change.tmpf = [axisref[x] for x in gain_change.freq]

Fig = Figure(size=(1000,600)) 
ax = Axis(Fig[1,1],title="Yaw",xticklabelsvisible=false,ylabel="Gain Change Mulitple")
barplot!(ax,gain_change.tmpf, gain_change.yaw_change,color=:green)
hlines!(ax,[1],linestyle=:dash, color=:black,linewidth=3)
errorbars!(ax,gain_change.tmpf,gain_change.yaw_change,gain_change.yaw_change_sem,color=:black,linewidth=3)
ax2 = Axis(Fig[2,1],title="Fx",ylabel="Gain Change Mulitple")
barplot!(ax2,gain_change.tmpf, gain_change.fx_change,color=:orchid)
hlines!(ax2,[1],linestyle=:dash, color=:black,linewidth=3)
errorbars!(ax2,gain_change.tmpf,gain_change.fx_change,gain_change.fx_change_sem,color=:black,linewidth=3)
ax.limits=(nothing,nothing,0,2)
ax2.limits=(nothing,nothing,0,2)
ax2.xticks = ([range(1,18,length=18)...],[string.(freqs)...])
ax2.xlabel="Frequency (Hz)"
hlines!(ax,mean(gain_change.yaw_change),linestyle=:dash,color=:firebrick,linewidth=4)
hlines!(ax2,mean(gain_change.fx_change),linestyle=:dash,color=:firebrick,linewidth=4)
save("Figs/NewTrackingFigs/GainChangeMultiple.png",Fig,px_per_unit=4)
Fig
##
by_moth = combine(groupby(all_data,[:moth,:freq])) do gdf
    (
        yaw_change = gdf[gdf.condition.=="post",:gain_yaw][1] / gdf[gdf.condition.=="pre",:gain_yaw][1],
        fx_change = gdf[gdf.condition.=="post",:gain_fx][1] / gdf[gdf.condition.=="pre",:gain_fx][1],
    )
end
mean_moth = combine(groupby(by_moth,[:moth])) do gdf 
    (
        yaw = mean(gdf.yaw_change),
        fx = mean(gdf.fx_change),
    )
end
leftjoin!(mean_moth,mc,on=:moth)
sort!(mean_moth,:mean_fz)
f = Figure()
ax = Axis(f[1,1],xlabel="Relative Change in Fz",ylabel = "Mean Change In Gain")
scatter!(ax,mean_moth.mean_fz, mean_moth.yaw,label="Yaw",color=:green,markersize=15)
scatter!(ax,mean_moth.mean_fz,mean_moth.fx,label="Fx",color=:orchid,markersize=15)
axislegend(ax,position=:lt)
save("Figs/NewTrackingFigs/GainFZBoth.png",f,px_per_unit=4)
f
##3
# quick check on if roll is tracking 
m = "2024_11_05"
trial = "pre"
fy = allmoths[m]["ft"*trial[1:3]][:,5]
ff = fft(fy)
fr = fftfreq(length(fy),Int(1e4))

f = Figure()
ax = Axis(f[1,1],xscale=log10)
lines!(ax, fr[2:200],abs.(ff)[2:200])
vlines!(ax,freqqs,linestyle=:dash,alpha=0.3)
f
##
GLMakie.to_font("Dejavu")
