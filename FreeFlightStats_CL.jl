using CairoMakie
using FFTW 
using CSV 
using JLD2 
using Statistics 
using DataFrames
using LinearAlgebra
using DSP 
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
##
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
##
pre = DataFrame() 
post = DataFrame() 
for (i,f) in enumerate(freqqs)
    mg,glow,ghigh,mp,std_p = tf_stats(Hi[i,:])
    tmp = Dict(
        "freq" => f, 
        "mg" => mg,
        "glow" => glow,
        "ghigh" => ghigh,
        "mp" => mp,
        "stp" => std_p
    )
    push!(pre,tmp,cols=:union)
    mg,glow,ghigh,mp,std_p = tf_stats(Hf[i,:])
    tmp = Dict(
        "freq" => f, 
        "mg" => mg,
        "glow" => glow,
        "ghigh" => ghigh,
        "mp" => mp,
        "stp" => std_p
    )
    push!(post,tmp,cols=:union)
end
pre.mp = unwrap(pre.mp)
post.mp = unwrap(post.mp)
CSV.write("Steven/CL_pre_stats.csv",pre)
CSV.write("Steven/CL_post_stats.csv",post)
##
F = Figure(size=(800,800)) 
ax = Axis(F[2,1],ylabel="Gain",xscale=log10,
    xticklabelsvisible=false,yticks=([0,0.1,1],[L"0",L"0.1",L"1"]),yscale=log10,yticklabelsize=25,
    ylabelsize=30)

lines!(ax,pre.freq,pre.mg,color=:steelblue,linewidth=3)
lines!(ax,post.freq,post.mg,color=:firebrick,linewidth=3)
band!(ax,pre.freq,pre.glow,pre.ghigh,color=:steelblue,alpha=0.3)
band!(ax,post.freq,post.glow,post.ghigh,color=:firebrick,alpha=0.3)

ax2 = Axis(F[3,1],ylabel="Phase (radians)",xlabel="Frequency (Hz)",
    xscale=log10,xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),limits=(0.1,nothing,nothing,nothing),
    yticks = ([0,-pi,-2pi, -3pi], [L"0",L"-\pi",L"-2\pi",L"-3\pi"]),yticklabelsize=25,xticklabelsize=25,
    ylabelsize=30,xlabelsize=30)

l = lines!(ax2,pre.freq,pre.mp,color=:steelblue,linewidth=3)
h = lines!(ax2,post.freq,post.mp,color=:firebrick,linewidth=3)
band!(ax2,pre.freq,pre.mp .- pre.stp,pre.mp .+ pre.stp,color=:steelblue,alpha=0.3)
band!(ax2,post.freq,post.mp .- post.stp,post.mp .+ post.stp,color=:firebrick,alpha=0.3)
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

meanpre = mean(epre,dims=2)[:]
meanpost = mean(epost,dims=2)[:]

stdpre = std(epre,dims=2)[:]
stdpost = std(epost,dims=2)[:]  

lowpre = meanpre .- stdpre ./ sqrt(size(Hi,2))
highpre = meanpre .+ stdpre ./ sqrt(size(Hi,2))
lowpost = meanpost .- stdpost ./ sqrt(size(Hf,2))
highpost = meanpost .+ stdpost ./ sqrt(size(Hf,2))


F = Figure(size=(800,400))
ax = Axis(F[1,1],xscale=log10,
    xlabel="Frequency (Hz)",xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),xlabelsize=30,xticklabelsize=25,
    ylabel = "Tracking Error",yticks=([0,1,2],[L"0",L"1",L"2"]), yticklabelsize=25,ylabelsize=30,
    limits=(0.1,nothing,nothing,nothing))
lines!(ax,freqqs,meanpre,linewidth=3,color=:steelblue)
band!(ax,freqqs,lowpre,highpre,color=:steelblue,alpha=0.3)
lines!(ax,freqqs,meanpost,linewidth=3,color=:firebrick)
band!(ax,freqqs,lowpost,highpost,color=:firebrick,alpha=0.3)
save("Figs/PaperFigs/Tracking_error.svg",F,px_per_unit=4)
F