using DataFrames
using JLD2 
using StatsBase
using FFTW
using SpecialFunctions
using LinearAlgebra
using Pipe 
using CairoMakie
using DataFramesMeta
using Interpolations
using DSP
using GLM
include("me_functions.jl")
include("stats_functions.jl")
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moths_set_1.jld" 
fs = Int(1e4)
N = Int(1e5)
freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
pixel_conversion = 0.14 ## mm/pixels 

##
mc = get_mean_changes(allmoths; axis=6)
moths = mc[mc.mean_gain .> 25,:moth]

Hi_fx = zeros(ComplexF64,18,length(moths))
Hf_fx = zeros(ComplexF64,18,length(moths))
Hi_yaw = zeros(ComplexF64,18,length(moths))
Hf_yaw = zeros(ComplexF64,18,length(moths))
Hi_roll = zeros(ComplexF64,18,length(moths))
Hf_roll = zeros(ComplexF64,18,length(moths))

for (i,m) in enumerate(moths)
    dat = allmoths[m]["ftpre"]
    datf = allmoths[m]["ftpos"]

    fpre = (allmoths[m]["stimpre"] .- mean(allmoths[m]["stimpre"])) .* pixel_conversion
    fpost = (allmoths[m]["stimpost"] .- mean(allmoths[m]["stimpost"])) .* pixel_conversion

    Hi_fx[:,i] = tf_freq(fpre,dat[:,1],freqqs,fs)
    Hf_fx[:,i] = tf_freq(fpost,datf[:,1],freqqs,fs)

    Hi_yaw[:,i] = tf_freq(fpre,dat[:,6],freqqs,fs)
    Hf_yaw[:,i] = tf_freq(fpost,datf[:,6],freqqs,fs)
    Hi_roll[:,i] = tf_freq(fpre,dat[:,5],freqqs,fs)
    Hf_roll[:,i] = tf_freq(fpost,datf[:,5],freqqs,fs)
end
##
pre_fx, post_fx = get_all_tf_stats(Hi_fx,Hf_fx,freqqs; freq_max=8)

##

f = Figure(size=(600,800)) 

scale_factor = 1e-4

ax = Axis(f[1,1],
    xscale = log10,
    yscale = log10,

    ylabel = "Gain",
    ylabelsize = 30,
    yticklabelsize = 25,

    yticks = ([1e-4, 1.5e-4, 5e-4],[L"1.0", L"1.5", L"5.0"]),
    # ytickformat = ys -> L"%$(round.(ys ./ scale_factor, digits=2))",

    limits = (0.1, 10, 5e-5, 6e-4),
    xticklabelsvisible = false,
    title = "Side Slip"
)
text!(
    ax.scene,
    L"\times 10^{-4}",
    position = Point2f(0.1, 5.5e-4),   # top-left of axis
    align = (:left, :center),
    # space = :relative,
    fontsize = 22
)
lines!(ax,pre_fx.freq,pre_fx.mg,color=:steelblue,linewidth=3)
lines!(ax,post_fx.freq,post_fx.mg,color=:firebrick,linewidth=3)
band!(ax,pre_fx.freq,pre_fx.glow,pre_fx.ghigh,color=:steelblue,alpha=0.3)
band!(ax,post_fx.freq,post_fx.glow,post_fx.ghigh,color=:firebrick,alpha=0.3)
ax2 = Axis(f[2,1],xscale=log10,
    ylabel="Phase (radians)",
    ylabelsize=30, yticklabelsize=25,
    limits=(0.1,10,nothing,nothing),
    yticks = ([-4pi,-3pi,-2pi,-pi,0], [L"-4\pi", L"-3\pi", L"-2\pi", L"-\pi", L"0"]),
    xlabel="Frequency (Hz)",
    xlabelsize=30,
    xticklabelsize=25,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"])
    )
lines!(ax2,pre_fx.freq,pre_fx.mp,color=:steelblue,linewidth=3)
lines!(ax2,post_fx.freq,post_fx.mp,color=:firebrick,linewidth=3)
band!(ax2,pre_fx.freq,pre_fx.mp .- pre_fx.stp,pre_fx.mp .+ pre_fx.stp,color=:steelblue,alpha=0.3)
band!(ax2,post_fx.freq,post_fx.mp .- post_fx.stp,post_fx.mp .+ post_fx.stp,color=:firebrick,alpha=0.3)
save("Figs/PaperFigs/FX_tracking.png",f,px_per_unit=4)
f 
##
pre_yaw, post_yaw = get_all_tf_stats(Hi_yaw,Hf_yaw,freqqs; freq_max=8)

f = Figure(size=(600,800))
ax = Axis(f[1,1],xscale=log10,
    ylabel="Gain",yscale=log10,
    ylabelsize=30, yticklabelsize=25,
    yticks = ( [10^-3,10^-2,10^-1] , [L"10^{-3}",L"10^{-2}",L"10^{-1}"]),
    limits=(0.1,10,0.0005,0.015),
    xticklabelsvisible=false,title = "Yaw"
    )
lines!(ax,pre_yaw.freq,pre_yaw.mg,color=:steelblue,linewidth=3)
lines!(ax,post_yaw.freq,post_yaw.mg,color=:firebrick,linewidth=3)
band!(ax,pre_yaw.freq,pre_yaw.glow,pre_yaw.ghigh,color=:steelblue,alpha=0.3)
band!(ax,post_yaw.freq,post_yaw.glow,post_yaw.ghigh,color=:firebrick,alpha=0.3)
ax2 = Axis(f[2,1],xscale=log10,
    ylabel="Phase (radians)",
    ylabelsize=30, yticklabelsize=25,
    limits=(0.1,10,-22,pi),
    yticks = ([-6pi,-4pi,-2pi,0], [L"-6\pi", L"-4\pi", L"-2\pi", L"0"]),
    xlabel="Frequency (Hz)",
    xlabelsize=30,
    xticklabelsize=25,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"])
    )
lines!(ax2,pre_yaw.freq,pre_yaw.mp,color=:steelblue,linewidth=3)
lines!(ax2,post_yaw.freq,post_yaw.mp,color=:firebrick,linewidth=3)
band!(ax2,pre_yaw.freq,pre_yaw.mp .- pre_yaw.stp,pre_yaw.mp .+ pre_yaw.stp,color=:steelblue,alpha=0.3)
band!(ax2,post_yaw.freq,post_yaw.mp .- post_yaw.stp,post_yaw.mp .+ post_yaw.stp,color=:firebrick,alpha=0.3)

save("Figs/PaperFigs/Yaw_tracking.png",f,px_per_unit=4)
f
##
pre_roll, post_roll = get_all_tf_stats(Hi_roll,Hf_roll,freqqs; freq_max=8)
f = Figure(size=(600,800))
ax = Axis(f[1,1],xscale=log10,
    ylabel="Gain",yscale=log10,
    ylabelsize=30, yticklabelsize=25,
    yticks = ( [0.005,0.01,0.015],[L"5",L"10",L"15"]),
    # yticks = ( [00,10^-1,10^0] , [L"10^{-2}",L"10^{-1}",L"10^{0}"]),
    limits=(0.1,10,0.003,0.02),
    xticklabelsvisible=false,title = "Roll"
    )
text!(
    ax.scene,
    L"\times 10^{-3}",
    position = Point2f(0.1, 0.018),   # top-left of axis
    align = (:left, :center),
    # space = :relative,
    fontsize = 22
)
lines!(ax,pre_roll.freq,pre_roll.mg,color=:steelblue,linewidth=3)
lines!(ax,post_roll.freq,post_roll.mg,color=:firebrick,linewidth=3)
band!(ax,pre_roll.freq,pre_roll.glow,pre_roll.ghigh,color=:steelblue,alpha=0.3)
band!(ax,post_roll.freq,post_roll.glow,post_roll.ghigh,color=:firebrick,alpha=0.3)
ax2 = Axis(f[2,1],xscale=log10,
    ylabel="Phase (radians)",
    ylabelsize=30, yticklabelsize=25,
    limits=(0.1,10,nothing,pi),
    yticks = ([-4pi,-2pi,0], [L"-4\pi", L"-2\pi", L"0"]),
    xlabel="Frequency (Hz)",
    xlabelsize=30,
    xticklabelsize=25,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"])
    )
lines!(ax2,pre_roll.freq,pre_roll.mp,color=:steelblue,linewidth=3)
lines!(ax2,post_roll.freq,post_roll.mp,color=:firebrick,linewidth=3)
band!(ax2,pre_roll.freq,pre_roll.mp .- pre_roll.stp,pre_roll.mp .+ pre_roll.stp,color=:steelblue,alpha=0.3)
band!(ax2,post_roll.freq,post_roll.mp .- post_roll.stp,post_roll.mp .+ post_roll.stp,color=:firebrick,alpha=0.3)
save("Figs/PaperFigs/Roll_tracking.png",f,px_per_unit=4)
f
##
p = angle.(Hi_yaw)
f = Figure() 
ax = Axis(f[1,1],xlabel="Frequency (Hz)",ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=15,
    xticklabelsize=15,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"])
    )
lines!(ax,freqqs,unwrap_negative(p[:,1]),color=:steelblue,linewidth=2)
lines!(ax,freqqs,unwrap_negative(p[:,2]),color=:green,linewidth=2)
lines!(ax,freqqs,unwrap_negative(p[:,3]),color=:orange,linewidth=2)
lines!(ax,freqqs,unwrap_negative(p[:,4]),color=:red,linewidth=2)
lines!(ax,freqqs,unwrap_negative(p[:,5]),color=:purple,linewidth=2)
f
## Stats? 
ms = repeat(moths, inner = 15)
fs = repeat(freqqs[1:15], outer=7)
his = hcat(Hi_fx[1:15,:]...)
angle_low = angle.(Hi_fx[1:15,:])
for i in 1:size(Hi_fx,2)
    angle_low[:,i] = unwrap_negative(angle_low[:,i])
end

lowmass_fx = DataFrame(
    moth = ms, 
    freq = fs, 
    gain = abs.(vec(his)),
    phase = vec(angle_low)
)

lowmass_fx.condition .= "Low Mass"
hfs = hcat(Hf_fx[1:15,:]...)
angle_high = angle.(Hf_fx[1:15,:])
for i in 1:size(Hf_fx,2)
    angle_high[:,i] = unwrap_negative(angle_high[:,i])
end

highmass_fx = DataFrame(
    moth = ms,
    freq = fs,
    gain = abs.(vec(hfs)),
    phase = vec(angle_high)
)
highmass_fx.condition .= "High Mass"
all_data_fx = vcat(lowmass_fx, highmass_fx)

his = hcat(Hi_yaw[1:15,:]...)
angle_low = angle.(Hi_yaw[1:15,:])
for i in 1:size(Hi_yaw,2)
    angle_low[:,i] = unwrap_negative(angle_low[:,i])
end

lowmass_yaw = DataFrame(
    moth = ms, 
    freq = fs, 
    gain = abs.(vec(his)),
    phase = vec(angle_low)
)

lowmass_yaw.condition .= "Low Mass"
hfs = hcat(Hf_yaw[1:15,:]...)
angle_high = angle.(Hf_yaw[1:15,:])
for i in 1:size(Hf_yaw,2)
    angle_high[:,i] = unwrap_negative(angle_high[:,i])
end

highmass_yaw = DataFrame(
    moth = ms,
    freq = fs,
    gain = abs.(vec(hfs)),
    phase = vec(angle_high)
)
highmass_yaw.condition .= "High Mass"
all_data_yaw = vcat(lowmass_yaw, highmass_yaw)


his = hcat(Hi_roll[1:15,:]...)
angle_low = angle.(Hi_roll[1:15,:])
for i in 1:size(Hi_roll,2)
    angle_low[:,i] = unwrap_negative(angle_low[:,i])
end

lowmass_roll = DataFrame(
    moth = ms, 
    freq = fs, 
    gain = abs.(vec(his)),
    phase = vec(angle_low)
)

lowmass_roll.condition .= "Low Mass"
hfs = hcat(Hf_roll[1:15,:]...)
angle_high = angle.(Hf_roll[1:15,:])
for i in 1:size(Hf_roll,2)
    angle_high[:,i] = unwrap_negative(angle_high[:,i])
end

highmass_roll = DataFrame(
    moth = ms,
    freq = fs,
    gain = abs.(vec(hfs)),
    phase = vec(angle_high)
)
highmass_roll.condition .= "High Mass"
all_data_roll = vcat(lowmass_roll, highmass_roll)
##
""" Models on Gain"""
##
gain_fx_lm = lm(@formula(gain ~ condition + freq), all_data_fx)
gain_yaw_lm = lm(@formula(gain ~ condition + freq), all_data_yaw)
gain_roll_lm = lm(@formula(gain ~ condition + freq), all_data_roll)
##
""" Models on Phase"""
##
phase_fx_lm = lm(@formula(phase ~ condition + freq), all_data_fx)
phase_yaw_lm = lm(@formula(phase ~ condition + freq), all_data_yaw)
phase_roll_lm = lm(@formula(phase ~ condition + freq), all_data_roll)
##

