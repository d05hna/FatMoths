using FFTW
using Statistics 
using DSP 
using LinearAlgebra 
using JLD2 
using GLMakie 
using AlgebraOfGraphics
using Pipe
using DataFramesMeta
using SpecialFunctions
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)
include("me_functions.jl")
include("stats_functions.jl")
fs = Int(1e2)
N = Int(2e3)
df = 0.05
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]

@load "fat_moth_free_flight.jld" 
##
"""
Significant Tracking Test per Simon's Science paper in 2015
Pipeline: 
    PSD for the Transfer Function of Moth and Flower
    Isolate Frequency band +- 0.5 Hz around the driving frequencies
        10 samples 
    Fishers Exact G test for periodicity for this band 
        confidence threshold of 0.05 
    Doesn't Work well below 3 Hz but that is fine because tracking is clear in that band 
        doesn't work bc things overlap 
"""
## Start with Closed Loop 
freqqs_to_test = freqqs[freqqs .>= 2.8]
sig_tracking_CL = DataFrame() 
for moth in collect(keys(FreeFlight))
    pre = FreeFlight[moth]["moth_pre"] 
    post = FreeFlight[moth]["moth_post"]

    for f in freqqs_to_test
        pval_pre = fisher_test_tracking(pre,fs,f,win=10)
        pval_post = fisher_test_tracking(post,fs,f,win=10)
        tmp = Dict(
            "Moth" => moth,
            "Frequency" => f,
            "pval" => pval_pre,
            "Trial" => "pre"
        )
        push!(sig_tracking_CL, tmp, cols=:union)

        tmp = Dict(
            "Moth" => moth,
            "Frequency" => f,
            "pval" => pval_post,
            "Trial" => "post"
        )
        push!(sig_tracking_CL, tmp, cols=:union)
    end
end

# How Many Moths have sig tracking per freq? 
sig_tracking_CL[!,:Significant] = sig_tracking_CL.pval .< 0.05

sig_tracking_CL.sig = ifelse.(sig_tracking_CL.Significant .== true, 1, 0)

##
stcl_pre = @pipe sig_tracking_CL |>
    @subset(_, :Trial .== "pre") |>
    groupby(_, :Frequency) |>
    @combine(_, :Frequency = first(:Frequency),
        :SigCount = sum(:sig),
        :Total = length(:sig),
        :Proportion = sum(:sig) / length(:sig)
    )

stcl_post = @pipe sig_tracking_CL |>
    @subset(_, :Trial .== "post") |>
    groupby(_, :Frequency) |>
    @combine(_, :Frequency = first(:Frequency),
        :SigCount = sum(:sig),
        :Total = length(:sig),
        :Proportion = sum(:sig) / length(:sig)
    )
##
F = Figure(size=(800,400))
ax = Axis(F[1,1],ylabel="Proportion Significant (N = 10)",xscale=log10,yticks=0:0.2:1,xticks=[3,5,8,10],
    xlabel="Frequency (Hz)",title="Significant Tracking Closed Loop")
lines!(ax,stcl_pre.Frequency,stcl_pre.Proportion,color=:steelblue,linewidth=3,label="Pre")
lines!(ax,stcl_post.Frequency,stcl_post.Proportion,color=:firebrick,linewidth=3,label="Post")
axislegend(ax)
save("Figs/SigTracking/sig_tracking_CL.png",F,px_per_unit=4)
F
## What moth ? 
stcl_pre_moth = @pipe sig_tracking_CL |>
    @subset(_, :Trial .== "pre") |>
    groupby(_, :Moth) |>
    @combine(_, :Moth = first(:Moth),
        :SigCount = sum(:sig),
        :Total = length(:sig),
        :Proportion = sum(:sig) / length(:sig)
    )
##
"""
Looks like moth_8 cant track at all above 3 hz wow 
""" 

x = FreeFlight["moth_1"]["moth_pre"]
ft = fft(x) 
fr = fftfreq(N,fs)
x2 = FreeFlight["moth_8"]["moth_pre"]
ft2 = fft(x2)
fig = Figure() 
ax = Axis(fig[1,1],title="Moth 8 Pre CL FFT",xscale=log10,xticks=[0.1,1,3,10])
lines!(ax,fr[2:200],abs.(ft)[2:200] ./N ,color=:steelblue,linewidth=3,label="Moth_1")
lines!(ax,fr[2:200],abs.(ft2)[2:200] ./N ,color=:firebrick,linewidth=3,label="Moth_8")
axislegend(ax)
vlines!(ax,freqqs,color=:grey,alpha=0.5)
fig