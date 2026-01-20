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
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)
include("me_functions.jl")
include("stats_functions.jl")
fs = Int(1e4)
N = Int(1e5)
df = 0.1
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]

@load "fat_moths_set_1.jld" 
##
mc = get_mean_changes(allmoths,axis=6)
moths = mc[mc.mean_gain .> 25,:moth]


ftnames = ["fx","fy","fz","tx","ty","tz"]
freqqs_to_test = freqqs[freqqs .>= 2.8] 

sig_tracking_tethered = DataFrame() 
for f in freqqs_to_test
    for m in moths
        pre = allmoths[m]["ftpre"]
        post = allmoths[m]["ftpos"]

        for (i, name) in enumerate(ftnames)
            signal_pre = pre[:,i]
            signal_post = post[:,i]

            pval_pre = fisher_test_tracking(signal_pre,fs,f,win=5)
            pval_post = fisher_test_tracking(signal_post,fs,f,win=5)

            tmp = Dict(
                "Moth" => m,
                "Frequency" => f,
                "Axis" => name,
                "pval" => pval_pre,
                "Trial" => "pre"
            )
            push!(sig_tracking_tethered, tmp, cols=:union)

            tmp = Dict(
                "Moth" => m,
                "Frequency" => f,
                "Axis" => name,
                "pval" => pval_post,
                "Trial" => "post"
            )
            push!(sig_tracking_tethered, tmp, cols=:union)
        end
    end
end
##
m_dict = Dict() 
for (i,m) in enumerate(moths)
    m_dict[m] = "moth_$i"

end
sig_tracking_tethered.Moth = [m_dict[m] for m in sig_tracking_tethered.Moth]
##
plot = data(sig_tracking_tethered) * mapping(:Frequency,:pval, row = :Moth,
    col=:Axis, color=:Trial) * visual(Scatter)  

fig = draw(plot,figure = (; size=(1000,1000)) ,axis=(; xscale=log10, xticks = [3,5,10]))
save("Figs/SigTracking/sig_tracking_tethered.png", fig)
fig 