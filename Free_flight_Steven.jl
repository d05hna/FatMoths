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
using MultivariateStats
using MAT
using FFTW
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
include("me_functions.jl")
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu-Bold")
theme.font = :bold
theme.fontcolor=:black
set_theme!(theme)



moth_names = ["091815_run1","100515_run2","102015_run1","102815_run3","021116_run3","080416_run2","080916_run2","080916_run5","092116_run2","111816_run1"]
d = read(matopen("seteven_data_pnt.mat"))

freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
##


## Calcualte The Stats
HI = DataFrame()
HF = DataFrame()
CI = DataFrame()
CF = DataFrame()
for i in 1:17
    hi = d["hi"][i,:]
    hf = d["hf"][i,:]
    ci = d["Ci"][i,:]
    cf = d["Cf"][i,:]

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
    push!(HI,tmp,cols=:union)
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
    push!(HF,tmp,cols=:union)
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
    push!(CI,tmp,cols=:union)
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
    push!(CF,tmp,cols=:union)
end
# HI.mean_phase = [x > 0 ? x - 2π : x for x in HI.mean_phase]
HI.mean_phase = unwrap(HI.mean_phase)
HI.p_lo = unwrap(HI.p_lo)
HI.p_hi = unwrap(HI.p_hi)
HF.mean_phase = unwrap(HF.mean_phase)
HF.p_lo = unwrap(HF.p_lo)
HF.p_hi = unwrap(HF.p_hi)
CI.mean_phase = unwrap(CI.mean_phase)
CF.mean_phase = unwrap(CF.mean_phase)
CI.p_lo = unwrap(CI.p_lo)
CI.p_hi = unwrap(CI.p_hi)
CF.p_lo = unwrap(CF.p_lo)
CF.p_hi = unwrap(CF.p_hi)

CI = CI[CI.freq .<= 10,:]
CF = CF[CF.freq .<= 10,:]
## Closed Loop 

f = Figure(size=(1200,800))
colgap!(f.layout, 20)
rowgap!(f.layout, 50)

ax = Axis(f[1,2],  ylabel="Gain",yscale=log10,xscale=log10,yticks=[0.1,1], ylabelfont=:bold,
    xticks = [0.2,1,5,10,20] ,   limits = (nothing,nothing,0.01,2),xticklabelsvisible=false)
lines!(ax,HI.freq,HI.mean_gain, color=:steelblue, label="Low Mass", linewidth=3)  
lines!(ax,HF.freq,HF.mean_gain,color = :firebrick, label="High Mass", linewidth=3) 
errorbars!(ax,HI.freq,HI.mean_gain,HI.g_lo,HI.g_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax,HF.freq,HF.mean_gain,HF.g_lo,HF.g_hi,color=:firebrick,whiskerwidth=10)

ax2 = Axis(f[2,2], xscale=log10,xticks = [0.1,1,10],xlabel="Frequency (Hz)",limits=(0.1,nothing,nothing,nothing),
    ylabel="Phase",ylabelfont=:bold,xlabelfont=:bold)
lines!(ax2,HI.freq,HI.mean_phase, color=:steelblue, label="Low Mass", linewidth=3)  
lines!(ax2,HF.freq,HF.mean_phase,color = :firebrick, label="High Mass", linewidth=3) 
errorbars!(ax2,HI.freq,HI.mean_phase,HI.p_lo,HI.p_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax2,HF.freq,HF.mean_phase,HF.p_lo,HF.p_hi,color=:firebrick,whiskerwidth=10)
ax2.yticks = ([0,-pi,-2pi],["0","-π","-2π"])
linkxaxes!(ax2,ax)


ax3 = Axis(f[2,1],xlabel = "Frequency (Hz)", xlabelfont=:bold,xticks=[0.1,1,10],xscale=log10,
    limits=(0.1,nothing,nothing,nothing),ylabel="Tracking Error",ylabelfont=:bold)
lines!(ax3,HI.freq,HI.mean_error,color=:steelblue,linewidth=3)
lines!(ax3,HF.freq,HF.mean_error,color=:firebrick,linewidth=3)
errorbars!(ax3,HI.freq,HI.mean_error,HI.low_error,HI.high_error,color=:steelblue,whiskerwidth=10)
errorbars!(ax3,HF.freq,HF.mean_error,HF.low_error,HF.high_error,color=:firebrick,whiskerwidth=10)
Legend(f[3,1:2],ax,orientation=:horizontal)

save("Figs/FreeFlight/closed_loop_CI.png",f,px_per_unit=4)
# save("Figs/FreeFlight/closed_loop.svg",f)


f
## Open Loop Response 

ci = d["Ci"]
cf = d["Cf"]

open_loop_i = DataFrame(freqs = freqqs,mean=abs.(mean(ci,dims=2))[:],sem = abs.(std(ci,dims=2) ./ sqrt(10))[:],angle = (angle.(mean(ci,dims=2))[:]) .- pi,angle_sem = (std(angle.(ci),dims=2) ./ sqrt(10))[:])
open_loop_i.condition .= "Low Mass"
open_loop_f = DataFrame(freqs = freqqs,mean=abs.(mean(cf,dims=2))[:],sem = abs.(std(cf,dims=2) ./ sqrt(10))[:],angle = (angle.(mean(cf,dims=2))[:]) .- pi,angle_sem = (std(angle.(cf),dims=2) ./ sqrt(10))[:])
open_loop_f.condition .= "High Mass"

open_loop = vcat(open_loop_i,open_loop_f,cols=:union)

mult = abs.((cf) ./ (ci))
mean_mult = mean(mult,dims=2)[:]
sem_mult = std(mult,dims=2)[:] ./ sqrt(10)

logx = log10.(freqqs[1:16])
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
widths = right .- left
##
f = Figure() 
ax = Axis(f[1,1],xscale=log10,limits=(0.15,11,0,4))
barplot!(ax,freqqs[1:16],mean_mult[1:16],width=widths,color=:grey)
errorbars!(ax,freqqs[1:16],mean_mult[1:16],sem_mult[1:16],color=:black)
lines!(ax, range(0.15,11,length=16),repeat([1.7],16),linestyle=:dash,color=:red)
f

ff = freqqs[1:16]
gpre = open_loop_i.mean[1:16]
gpo = open_loop_f.mean[1:16]
sempre = open_loop_i.sem[1:16]
sempo  = open_loop_f.sem[1:16]
apre = open_loop_i.angle[1:16]
apo = open_loop_f.angle[1:16]
apresem = open_loop_i.angle_sem[1:16]
apossem = open_loop_f.angle_sem[1:16]
## Big Open Loop Fig 

F = Figure(size=(1200,600))
ax1 = Axis(F[1,1],xlabelfont=:bold,ylabel="Gain",ylabelfont=:bold,yticks=[10^-2,10^-1,10^0],limits=(nothing,10,nothing,nothing),
    xscale=log10,yscale=log10,xticklabelsvisible=false)
lines!(ax1,CI.freq,CI.mean_gain,color=:steelblue,linewidth=3,label = "Low Mass")
lines!(ax1,CF.freq,CF.mean_gain,color=:firebrick,linewidth=3,label = "High Mass")
errorbars!(ax1,CI.freq,CI.mean_gain,CI.g_lo,CI.g_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax1,CF.freq,CF.mean_gain,CF.g_lo,CF.g_hi,color=:firebrick,whiskerwidth=10)

ax2 = Axis(F[2,1],xlabel = "Frequency (Hz)",xlabelfont=:bold,ylabel="Phase",ylabelfont=:bold,yticks=([0,-pi,-2pi],["0","-π","-2π"]),limits=(0.19,10,-2pi,nothing),xscale=log10
    )
lines!(ax2,CI.freq,CI.mean_phase,color=:steelblue,linewidth=3,label = "Low Mass")
lines!(ax2,CF.freq,CF.mean_phase,color=:firebrick,linewidth=3,label = "High Mass")
errorbars!(ax2,CI.freq,CI.mean_phase,CI.p_lo,CI.p_hi,color=:steelblue,whiskerwidth=10)
errorbars!(ax2,CF.freq,CF.mean_phase,CF.p_lo,CF.p_hi,color=:firebrick,whiskerwidth=10)
linkxaxes!(ax2,ax1)
ax2.xticks=[1,10]

ax3 = Axis(F[1,2],xlabel = "Frequency (Hz)",xlabelfont=:bold,ylabel="Gain Change Multiple",ylabelfont=:bold,yticks=[0,2,4],limits=(0.19,10,0,4),xscale=log10,
    xticks=[1,10]    )
barplot!(ax3,freqqs[1:16],mean_mult[1:16],width=widths,color=:grey)
errorbars!(ax3,freqqs[1:16],mean_mult[1:16],sem_mult[1:16],color=:black)
lines!(ax3, range(0.15,11,length=16),repeat([1.7],16),linestyle=:dash,color=:red)
Legend(F[3,1:2],ax1,orientation=:horizontal)
save("Figs/FreeFlight/open_loop_CI.png",F,px_per_unit=4)
F
##
"""
Stats For Anova: 
Gain, Phase, Error 
Within SUbject Frequency, COndition 
Interactions? Frequency X Condition 
Random Affect Subject 
""" 

using SimpleANOVA,CategoricalArrays
##
ghlo = abs.(d["Ci"])
ghhi = abs.(d["Cf"])
phlo = unwrap(angle.(d["Ci"]),dims=1)
phhi = unwrap(angle.(d["Cf"]),dims=1)
ehlo = tracking_error.(d["hi"])
ehhi = tracking_error.(d["hf"])

glo = DataFrame(ghlo,moth_names)
glo.freq = freqqs
glo = stack(glo,Not(:freq),variable_name="moth",value_name="gain")
glo.condition .= "low"
ghi = DataFrame(ghhi,moth_names)
ghi.freq = freqqs
ghi = stack(ghi,Not(:freq),variable_name="moth",value_name="gain")
ghi.condition .= "high"
g = vcat(glo,ghi)

g.condition = categorical(g.condition)
g.freq = categorical(g.freq)
g.moth = categorical(g.moth)

plo = DataFrame(phlo,moth_names)
plo.freq = freqqs
plo = stack(plo,Not(:freq),variable_name="moth",value_name="phase")
plo.condition .= "low"
phi = DataFrame(phhi,moth_names)
phi.freq = freqqs
phi = stack(phi,Not(:freq),variable_name="moth",value_name="phase")
phi.condition .= "high"
p = vcat(plo,phi)

p.condition = categorical(p.condition)
p.freq = categorical(p.freq)
p.moth = categorical(p.moth)


elo = DataFrame(ehlo,moth_names)
elo.freq = freqqs
elo = stack(elo,Not(:freq),variable_name="moth",value_name="error")
elo.condition .= "low"
ehi = DataFrame(ehhi,moth_names)
ehi.freq = freqqs
ehi = stack(ehi,Not(:freq),variable_name="moth",value_name="error")
ehi.condition .= "high"
e = vcat(elo,ehi)

e.condition = categorical(e.condition)
e.freq = categorical(e.freq)
e.moth = categorical(e.moth)

##
resg = anova(g, :gain, [:condition, :freq, :moth], [fixed, fixed, subject])
resp = anova(p,:phase,[:condition,:freq,:moth],[fixed,fixed,subject])
rese = anova(e,:error,[:condition,:freq,:moth],[fixed,fixed,subject])

##
