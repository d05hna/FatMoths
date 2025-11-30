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
##
mc = get_mean_changes(allmoths;axis=6)
moths = ["2024_11_01","2024_11_08","2024_11_11","2025_03_20","2025_04_02","2025_09_19","2025_10_10"]
n_moths = length(moths)
##
df = allmoths["2024_11_08"]["data"]
put_stim_in!(df,allmoths,"2024_11_11")
##
d = combine(groupby(df,[:moth,:wb,:muscle])) do gdf 
    (
    trial = gdf.trial[1] =="pre" ? 0 : 1,
    species=gdf.species[1],
    firsttime= minimum(gdf.time),
    firstphase = minimum(gdf.phase),
    count = length(gdf.time),
    wblen = mean(gdf.wblen),
    tz = mean(gdf.tz),
    stim = mean(gdf.pos)
    )
end 
##
num = length(unique(d.muscle))
count = select(d,:wb,:muscle,:count,:trial,:stim)
countmat = Matrix(dropmissing((select(unstack(count,:muscle,:count),Not(:wb)))))
time = select(d,:wb,:muscle,:firstphase,:stim)
timemat = Matrix(dropmissing(select(unstack(time,:muscle,:firstphase),Not(:wb))))
tris = Bool.(countmat[:,1])
stim = countmat[:,2]

fullmat = zscore(timemat[:,2:end])


##
pc = fit(MultivariateStats.PCA,fullmat')
lat = pc.proj' * fullmat'

 
##
F = Figure() 
ax = Axis3(F[1,1])
scatter!(ax,lat[1,:],lat[2,:],lat[3,:],color=stim)
F
##
dlm = d[d.muscle.=="ldlm" .|| d.muscle.=="rdlm",[:wb,:muscle,:firsttime,:trial,:stim]]
timemat = dropmissing(select(unstack(dlm,:muscle,:firsttime),Not(:wb)))
timemat.off = timemat.rdlm - timemat.ldlm 

##
F = Figure() 
ax = Axis(F[1,1])
scatter!(ax,timemat.off,color=timemat.stim)
F