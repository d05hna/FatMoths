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
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]

##
mc = get_mean_changes(allmoths;axis=6)
moths = mc[mc.mean_gain .> 25,:moth]
n_moths = length(moths)
##
bigdf = DataFrame() 
for m in moths 
    bigdf = vcat(bigdf,put_stim_in!(allmoths[m]["data"],allmoths,m),cols=:union)
end
##
d = combine(groupby(bigdf,[:moth,:wb,:muscle])) do gdf 
    (
    trial = gdf.trial[1] == "pre" ? 0 : 1, 
    species=gdf.species[1],
    firsttime= minimum(gdf.time),
    firstphase = minimum(gdf.phase),
    count = length(gdf.time),
    wblen = mean(gdf.wblen),
    tz = mean(gdf.tz),
    stim = mean(gdf.pos),
    fz = mean(gdf.fz)
    )
end 
d.trial = [x==0 ? "pre" : "post" for x in d.trial]


## Increase in Axillary Count Assymetry 
plot = data(d[d.muscle.=="lax" .|| d.muscle.=="rax",:])*mapping(:count,row=:muscle,color=:trial)*histogram(bins=10)*visual(alpha=0.8) |>draw
##
ax = d[d.muscle.=="lax" .|| d.muscle.=="rax",:]
long_ax = dropmissing(unstack(select(ax,Not(:firsttime,:firstphase)),:muscle,:count))
long_ax_time = dropmissing(unstack(select(ax,Not(:firsttime,:count)),:muscle,:firstphase))
long_ax_time.tdiff = long_ax_time.lax - long_ax_time.rax
long_ax.diff = long_ax.lax - long_ax.rax
plot = data(long_ax)*mapping(:diff,color=:trial)*histogram(bins=10)*visual(alpha=0.6)
plot2 = data(long_ax)*mapping(:tz,color=:trial)*histogram(bins=100)*visual(alpha=0.6)
f = Figure() 
ax1 = Axis(f[1,1])
draw!(ax1,plot)
ax2= Axis(f[1,2])
draw!(ax2,plot2)
f
ax3 = Axis(f[2,1:2])
theme.palette= (; colors = [:red,:orange,:blue,:green,:yellow,:purple,:black])
set_theme!(theme)
draw!(ax3,data(long_ax)*mapping(:diff,:tz,color=:moth)*visual(alpha=0.5)*linear()
)
f
##
long = unstack(select(d,Not(:firsttime,:count)),:muscle,:firstphase)
long = dropmissing!(select(long,Not(:rsa)))
sub = select(long,Not(:moth,:wb,:trial,:species,:wblen,:tz,:stim))
mat = Matrix(sub)' 

pc = fit(MultivariateStats.PCA,mat)
lat = pc.proj' * mat 

tmp = DataFrame("pc1"=> lat[1,:],"pc2" => lat[2,:],"pc3" => lat[3,:],"moth" => long.moth,"trial"=>long.trial,"wblen"=>long.wblen,"tz"=>long.tz)
plot = data(tmp)*mapping(:pc1,:pc2,color=:trial,row=:moth)|>draw
##
bigax = leftjoin(select(long_ax,Not(:lax,:rax)),select(long_ax_time,Not(:lax,:rax)),on=[:moth,:wb,:trial,:species,:tz,:stim,:wblen])
##
theme.palette= (; color = [:firebrick,:steelblue,:blue,:green,:yellow,:purple,:black])
set_theme!(theme)
plot = data(bigax[abs.(bigax.tdiff) .< 0.5,:])*mapping(:diff,:tdiff,layout=:moth,color=:trial)*linear()
plot2 = data(bigax[abs.(bigax.tdiff) .<0.5,:])*mapping(:diff,layout=:moth,color=:trial)*
    histogram(bins=10,normalization=:probability)*visual(alpha=0.7)|>draw
##
dlm = d[d.muscle.=="ldlm" .|| d.muscle.=="rdlm",:]
long_dlm = dropmissing(unstack(select(dlm,Not(:firsttime,:count)),:muscle,:firstphase))
long_dlm.diff = long_dlm.ldlm - long_dlm.rdlm
##
plot = data(long_dlm)*mapping(:diff,layout=:moth,color=:trial)*histogram(bins=100,normalization=:probability)|>draw
##
theme.palette= (; color=[:orchid,:forestgreen,:coral,:black])
set_theme!(theme)
d_phase = unstack(select(d,Not(:firsttime,:count)),:muscle,:firstphase)
sadlm = dropmissing(select(d_phase,Not(:lax,:lba,:ldvm,:rdvm,:rba,:rax)))

sadlm.ldiff = sadlm.ldlm - sadlm.lsa 
sadlm.rdiff = sadlm.rdlm - sadlm.rsa

plot = data(sadlm)*mapping(:rdiff,:fz,color=:moth,marker=:trial)
plot2 = data(sadlm)*mapping(:rdiff,color=:trial,layout=:moth)*histogram(bins=100,normalization=:probability)|>draw
##
pre = allmoths["2024_11_08"]["ftpre"][:,6]
post = allmoths["2024_11_08"]["ftpos"][:,6]

stimpre = allmoths["2024_11_08"]["stimpre"]
stimpost = allmoths["2024_11_08"]["stimpost"]

itp = interpolate(stimpre, BSpline(Linear()))
stimpre = itp(LinRange(1, length(itp), Int(1e5)))


itp = interpolate(stimpost, BSpline(Linear()))
stimpost = itp(LinRange(1, length(itp), Int(1e5)))


ftpre = fft(pre)
ftpost = fft(post)
fr = fftfreq(Int(1e5),Int(1e4))

fstimpre = fft(stimpre)
fstimpost = fft(stimpost)


h_pre = ftpre ./ fstimpre
h_post = ftpost ./ fstimpost

freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
f = Figure() 
ax = Axis(f[1,1],xscale=log10,ylabel="Amplitude",xlabel="Frequency")
lines!(ax,fr[2:200],abs.(h_pre)[2:200],label="Low Mass",color=:steelblue)
lines!(ax,fr[2:200],abs.(h_post)[2:200],label="High Mass",color=:firebrick)
# vlines!(ax,freqqs,color=:grey,label="Flower Frequency",alpha=0.3)
axislegend(ax)
f
##
tf_pre = tf_freq(stimpre .*0.14,pre.*1000,freqqs,Int(1e4))
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
f = Figure() 
ax = Axis(f[1,1],xscale=log10,ylabel="Gain (Yaw / Flower Position)",xlabel="Frequency",yscale=log10)
scatter!(ax,freqqs,abs.(tf_pre),label="Low Mass",color=:steelblue)
# vlines!(ax,freqqs,color=:grey,label="Flower Frequency",alpha=0.3)
ax2 = Axis(f[2,1],xscale=log10,ylabel="Phase (rad)",limits=(nothing,nothing,-2pi,0))
scatter!(ax2,freqqs,unwrap(angle.(tf_pre)),)
# axislegend(ax)
f