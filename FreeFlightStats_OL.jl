using CSV 
using DataFrames 
using MAT 
using DataFrames 
using CairoMakie 
using Statistics 
using LinearAlgebra
using JLD2
using DSP 
include("stats_functions.jl")
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moth_free_flight.jld" FreeFlight
pixel_conversion = 0.14 ## mm/pixels 


freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]

pnt = matread("Steven/steven_data_linear.mat")["results"]
Ci = hcat(pnt["Ci"]...)
Cf = hcat(pnt["Cf"]...)
##
pre = DataFrame() 
post = DataFrame() 

for (i,f) in enumerate(freqqs)
    mg,glow,ghigh,mp,p_std = tf_stats(Ci[i,:])
    tmp = Dict(
        "freq"  => f,
        "mg"    => mg,
        "glow"  => glow,
        "ghigh" => ghigh,
        "mp"    => mp,
        "stp"  => p_std,
    )
    push!(pre, tmp,cols=:union)
    mg,glow,ghigh,mp,p_std = tf_stats(Cf[i,:])
    tmp = Dict(
        "freq"  => f,
        "mg"    => mg,
        "glow"  => glow,
        "ghigh" => ghigh,
        "mp"    => mp,
        "stp"  => p_std,
    )
    push!(post, tmp,cols=:union)
end
pre.mp = unwrap(pre.mp)
post.mp = unwrap(post.mp)
CSV.write("FreeFlight_OL_stats_pre.csv",pre)
CSV.write("FreeFlight_OL_stats_post.csv",post)

## Per the Sig Tracking Results, Only keeping frequencies below 8.9 Hz
pre = pre[pre.freq .< 8.9,:]
post = post[post.freq .< 8.9,:]
##
F = Figure(size=(800,800)) 
ax = Axis(F[2,1],xscale = log10,xticklabelsvisible=false, 
    ylabel="Gain", yticks = ([0.01,0.1,1],[L"0.01",L"0.1",L"1"]),yscale=log10,
    ylabelsize=30, yticklabelsize=25,
    )

lines!(ax,pre.freq,pre.mg,color=:steelblue,linewidth=3)
lines!(ax,post.freq,post.mg,color=:firebrick,linewidth=3)
band!(ax,pre.freq,pre.glow,pre.ghigh,color=:steelblue,alpha=0.3)
band!(ax,post.freq,post.glow,post.ghigh,color=:firebrick,alpha=0.3)
ax2 = Axis(F[3,1],limits=(0.1,nothing,nothing,nothing),
    ylabel="Phase (radians)", yticks = ([0,-pi/2, -pi,-3pi/2], [L"0",L"-\frac{\pi}{2}",L"-\pi",L"-\frac{3\pi}{2}"]),
    ylabelsize=30, yticklabelsize=25,
    xlabel="Frequency (Hz)",xscale=log10,xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    xlabelsize=30, xticklabelsize=25,
    )

l = lines!(ax2,pre.freq,pre.mp,color=:steelblue,linewidth=3)
h = lines!(ax2,post.freq,post.mp,color=:firebrick,linewidth=3)
band!(ax2,pre.freq,pre.mp .- pre.stp,pre.mp .+ pre.stp,color=:steelblue,alpha=0.3)
band!(ax2,post.freq,post.mp .- post.stp,post.mp .+ post.stp,color=:firebrick,alpha=0.3)

linkxaxes!(ax2,ax)
Legend(F[1,1],[l,h],["Low Mass","High Mass"],orientation = :horizontal)
save("Figs/PaperFigs/OL_tracking.svg",F,px_per_unit=4)
F
##
gci = abs.(Ci)[1:15,:]
gcf = abs.(Cf)[1:15,:]

gchange = gcf ./ gci

mgchange = mean(gchange,dims=2)[:]
semchange = std(gchange,dims=2)[:] ./sqrt(10)

logx = log10.(freqqs[1:15])
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
widths = right .- left

f = Figure(size=(800,400)) 
ax = Axis(f[1,1],xscale=log10,limits=(0.15,10,0,4))
ax.xticks=[0.2,1,10]
ax.xlabel="Frequency (Hz)"
ax.ylabel = "Gain Change Multiple"
barplot!(ax,freqqs[1:15],mgchange,width=widths,color=:grey)
errorbars!(ax,freqqs[1:15],mgchange,semchange,color=:black)
lines!(ax, range(0.15,10,length=16),repeat([1.7],16),linestyle=:dash,color=:red,label="No Change")
axislegend(ax)
# save("Figs/PaperFigs/barplot_OL.svg",f,px_per_unit=4)
f