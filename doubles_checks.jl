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
using HypothesisTests
using MultivariateStats

include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_black()
theme.palette = (color = [:steelblue,:firebrick],)
theme.fontsize = 20
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
@load "fat_moths_set_1.jld" allmoths
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
##
all_data = DataFrame() 

for m in collect(keys(allmoths))
    tmp = allmoths[m]["data"]
    if "pos" in names(tmp)
    else
        put_stim_in!(tmp,allmoths,m)
    end
    all_data = vcat(all_data,tmp,cols=:union)
end
##
all_data = add_relpos_column!(all_data)

##
for m in unique(all_data.muscle)
    sub = all_data[all_data.muscle.==m,:]
    plot = data(sub)*mapping(:phase,color=:trial=>renamer(["pre"=>"Before Feeding","post"=> "After Feeding"]),
        row=:moth,col=:relpos)*histogram(bins=100,normalization=:probability)*visual(alpha=0.5)
    f = draw(plot,axis=(;))
    save("Relpos/$m.png",f)
end
    
##

tmp = unique(select(all_data,:wb,:trial,:moth,:pos,:vel,:wbtime))
tmp = combine(groupby(tmp,[:wb,:moth])) do gdf 
    (
            pos = gdf.pos[1],
            trial = gdf.trial[1],
            vel = gdf.vel[1],
            wbtime = gdf.wbtime[1]
        )
    end

firstspike = @pipe all_data |> 
    groupby(_,[:wb,:muscle,:moth]) |> 
    combine(_,:time=>minimum=>:time) |>
    unstack(_,[:wb,:moth],:muscle,:time)

# dropmissing!(firstspike)

leftjoin!(firstspike,tmp,on=[:wb,:moth])

##
tmp = dropmissing(select(firstspike,[:wb,:rsa,:moth,:trial]))
plot = data(tmp)*mapping(:wb,:rsa,col=:moth,row=:trial,color=:trial)
draw(plot, facet = (linkxaxes = false,))
##

lines(firstspike[firstspike.moth.=="2024_11_20",:wbtime],firstspike[firstspike.moth.=="2024_11_20",:lsa])
##

d4t = firstspike[firstspike.moth.=="2024_11_08",:]
dropmissing!(d4t)

spikesmat = Matrix(select(d4t,Not(:wb,:moth,:pos,:trial,:vel,:wbtime)))'
fmat = Matrix(select(d4t,:pos))'

m = fit(MultivariateStats.CCA,spikesmat,fmat)

##
lat = predict(m,spikesmat,:x)

stimfreq = round.(fftfreq(250,25),digits=4)
stimidx= [i for i in 1:250 if stimfreq[i] in freqqs]



coh_ldlm = mt_coherence(hcat(d4t[d4t.trial.=="pre",:lsa],d4t[d4t.trial.=="pre",:pos])';fs=25,nfft=250).coherence[1,2,stimidx]
scatter(coh_ldlm) 

