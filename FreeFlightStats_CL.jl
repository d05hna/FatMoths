using CairoMakie
using FFTW 
using CSV 
using JLD2 
using Statistics 
using DataFrames
using LinearAlgebra
using DSP 
using GLM
include("stats_functions.jl")
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moth_free_flight.jld" FreeFlight


freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
N = 2000 
fs = 100
## Get the transfer functions
Hi = zeros(18,10) |> x -> Complex.(x)
Hf = zeros(18,10) |> x -> Complex.(x)
fr = round.(fftfreq(N,fs),digits=4)
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
## Compute stats
pre,post,= get_all_tf_stats(Hi,Hf,freqqs; freq_max=8)
CSV.write("Steven/CL_pre_stats.csv",pre)
CSV.write("Steven/CL_post_stats.csv",post)
## CL Bode Plot 
F = Figure(size=(800,800)) 
ax = Axis(F[2,1],ylabel="Gain",xscale=log10,
    xticklabelsvisible=false,yticks=([0,0.1,1],[L"0",L"0.1",L"1"]),yscale=log10,yticklabelsize=25,
    ylabelsize=30,
    limits = (0.1,nothing,0.05,3))

lines!(ax,pre.freq,pre.mg,color=:steelblue,linewidth=4)
lines!(ax,post.freq,post.mg,color=:firebrick,linewidth=4)
band!(ax,pre.freq,pre.glow,pre.ghigh,color=:steelblue,alpha=0.5)
band!(ax,post.freq,post.glow,post.ghigh,color=:firebrick,alpha=0.5)

ax2 = Axis(F[3,1],ylabel="Phase (radians)",xlabel="Frequency (Hz)",
    xscale=log10,xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),limits=(0.1,nothing,nothing,nothing),
    yticks = ([0,-pi,-2pi, -3pi], [L"0",L"-\pi",L"-2\pi",L"-3\pi"]),yticklabelsize=25,xticklabelsize=25,
    ylabelsize=30,xlabelsize=30)

l = lines!(ax2,pre.freq,pre.mp,color=:steelblue,linewidth=4)
h = lines!(ax2,post.freq,post.mp,color=:firebrick,linewidth=4)
band!(ax2,pre.freq,pre.mp .- pre.stp,pre.mp .+ pre.stp,color=:steelblue,alpha=0.5)
band!(ax2,post.freq,post.mp .- post.stp,post.mp .+ post.stp,color=:firebrick,alpha=0.5)
linkxaxes!(ax2,ax)
Legend(F[1,1],[l,h],["Low Mass","High Mass"],orientation = :horizontal)
save("Figs/PaperFigs/CL_tracking.svg",F,px_per_unit=4)
F
## Tracking Error Plot 
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

meanpre = mean(epre,dims=2)[:]
meanpost = mean(epost,dims=2)[:]

stdpre = std(epre,dims=2)[:]
stdpost = std(epost,dims=2)[:]  

lowpre = meanpre .- stdpre ./ sqrt(size(Hi,2))
highpre = meanpre .+ stdpre ./ sqrt(size(Hi,2))
lowpost = meanpost .- stdpost ./ sqrt(size(Hf,2))
highpost = meanpost .+ stdpost ./ sqrt(size(Hf,2))

##
F = Figure(size=(800,400))
ax = Axis(F[1,1],xscale=log10,
    xlabel="Frequency (Hz)",xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),xlabelsize=30,xticklabelsize=25,
    ylabel = "Tracking Error",yticks=([0,1,2],[L"0",L"1",L"2"]), yticklabelsize=25,ylabelsize=30,
    limits=(0.1,nothing,nothing,nothing))
lines!(ax,freqqs,meanpre,linewidth=4,color=:steelblue)
band!(ax,freqqs,lowpre,highpre,color=:steelblue,alpha=0.5)
lines!(ax,freqqs,meanpost,linewidth=4,color=:firebrick)
band!(ax,freqqs,lowpost,highpost,color=:firebrick,alpha=0.5)
save("Figs/PaperFigs/Tracking_error.svg",F,px_per_unit=4)
F
## GLM for closed loop gain, phase, tracking error 
# put it all in a dataframe 
ms = ["moth_$i" for i in 1:10]
ms = repeat(ms, inner = 18)
fs = repeat(freqqs, outer=10)
his = hcat(Hi...)

lowmass = DataFrame(
    moth = ms, 
    freq = fs, 
    gain = abs.(vec(his)),
    phase = angle.(vec(his))
)

lowmass.condition .= "Low Mass"
hfs = hcat(Hf...)
highmass = DataFrame(
    moth = ms,
    freq = fs,
    gain = abs.(vec(hfs)),
    phase = angle.(vec(hfs))
)
highmass.condition .= "High Mass"
all_data = vcat(lowmass, highmass)
##
gain_lm = lm(@formula(gain ~ condition + freq), all_data)
phase_lm = lm(@formula(phase ~ condition + freq), all_data)

## Tracking Error DataFramea and stats 
epres = hcat(epre...)
eposts = hcat(epost...)
lowmass_error = DataFrame(
    moth = ms,
    freq = fs,
    error = vec(epres)
)
lowmass_error.condition .= "Low Mass"
highmass_error = DataFrame(
    moth = ms,
    freq = fs,
    error = vec(eposts)
)
highmass_error.condition .= "High Mass"
all_error = vcat(lowmass_error, highmass_error)

error_lm = lm(@formula(error ~ condition + freq), all_error)