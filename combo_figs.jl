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
using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:steelblue,:firebrick])
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
## sub out moths that dont increase the fx+tz + roll more than 25% on average 
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

    Yawi[:,i] = tf_freq(stimpre,syaw,freqqs,fs)s,:steelb
    Yawf[:,i] = tf_freq(stimpost,fyaw,freqqs,fs)
    Fxi[:,i] = tf_freq(stimpre,sfx,freqqs,fs)
    Fxf[:,i] = tf_freq(stimpost,ffx,freqqs,fs)
    Ci[:,i] = tf_freq(stimpre,ci,freqqs,fs)
    Cf[:,i] = tf_freq(stimpost,cf,freqqs,fs)
end
##
mci = mean(abs.(Ci),dims=2)[:]
mcf = mean(abs.(Cf),dims=2)[:]
sti = std(abs.(Ci),dims=2)[:] ./ sqrt(n_moths)
stf = std(abs.(Cf),dims=2)[:] ./ sqrt(n_moths)

pci = angle.(Ci)
pcf = angle.(Cf)
for i in 1:7
    pci[:,i] = unwrap_negative(pci[:,i])
    pcf[:,i] = unwrap_negative(pcf[:,i])

end

mpci = mean(pci,dims=2)[:]
mpcf = mean(pcf,dims=2)[:]

stpi = std(pci,dims=2)[:] ./ sqrt(n_moths)
stpf = std(pcf,dims=2)[:] ./ sqrt(n_moths)

##
f = Figure(size=(800,800)) 
ax = Axis(f[1,1],xscale=log10,yscale=log10,yticks=([0.01,0.1,1],[L"0.01",L"0.1",L"1"]),xticklabelsvisible=false,
limits=(nothing,nothing,0.01,2),ylabel="Gain")
lines!(ax,freqqs,myi,color=:steelblue,linewidth=3)
lines!(ax,freqqs,myf,color=:firebrick,linewidth=3)
errorbars!(ax,freqqs,myi,sti,whiskerwidth=10,color=:steelblue)
errorbars!(ax,freqqs,myf,stf,whiskerwidth=10,color=:firebrick)

ax2 = Axis(f[2,1],xscale=log10,xticks=[0.1,1,10],yticks=([0,-pi,-3pi,-5pi],[L"0",L"-\pi",L"-3\pi",L"-5\pi"]),
    ylabel="Phase (Radians)",xlabel="Frequency (Hz)",limits=(0.1,nothing,nothing,nothing))
lines!(ax2,freqqs,mpci,color=:steelblue,linewidth=3)
lines!(ax2,freqqs,mpcf,color=:firebrick,linewidth=3)
errorbars!(ax2,freqqs,mpci,stpi,color=:steelblue,whiskerwidth=10)
errorbars!(ax2,freqqs,mpcf,stpf,color=:firebrick,whiskerwidth=10)
linkxaxes!(ax,ax2)
save("Figs/PaperFigs/Tethered_Bode.svg",f)
f
##
alldata = DataFrame() 
for moth in moths 
    d = allmoths[moth]["data"]
    put_stim_in!(d,allmoths,moth)
    alldata = vcat(alldata,d,cols=:union)
end
##
df = combine(groupby(alldata,[:moth,:wb,:muscle])) do gdf 
    (
    trial = gdf.trial[1],
    species=gdf.species[1],
    firsttime= minimum(gdf.time),
    firstphase = minimum(gdf.phase),
    count = length(gdf.time),
    wblen = mean(gdf.wblen),
    tz = mean(gdf.tz),
    stim = mean(gdf.pos)
    )
end 
wide = unstack(select(df,Not(:firstphase,:count)),:muscle,:firsttime)
wide.lsaoff = wide.lsa - wide.ldlm 
wide.rsaoff = wide.rsa - wide.rdlm
wide.dlmoff = 1000 .*(wide.ldlm - wide.rdlm)


dlm = dropmissing(select(wide,:moth,:trial,:dlmoff,:stim,:wb))


dlm = combine(groupby(dlm, [:moth, :trial])) do sub
    sub.new_wb = sub.wb .- minimum(sub.wb)
    sub
end

plot = data(dlm[dlm.moth.!="2025_10_17",:])*mapping(:new_wb => "Wing Beat",:dlmoff => "L-R DLM Timing Offset",layout=:moth,color=:trial => renamer(["pre"=> "Low Mass","post"=>"High Mass"])=> "Body Condition")
f = draw(plot,figure=(; size=(800,800)))
save("Figs/GoodFigs/DLMoff_subset.png",f,px_per_unit=4)
f