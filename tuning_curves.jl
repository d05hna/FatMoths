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
using ColorSchemes
using Loess
# using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_minimal()
theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)

@load "fat_moths_set_1.jld" allmoths
pixel_conversion = 0.14 ## mm/pixels 
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]

##
mc = get_mean_changes(allmoths;axis=6)
moths = ["2024_11_01","2024_11_08","2024_11_11","2025_03_20","2025_04_02","2025_09_19","2025_10_10"]
n_moths = length(moths)
##
df = allmoths["2025_10_17"]["data"]
put_stim_in!(df,allmoths,"2025_10_17")
##
d = combine(groupby(df,[:moth,:wb,:muscle])) do gdf 
    (
    trial = gdf.trial[1] == "pre" ? 0 : 1, 
    species=gdf.species[1],
    firsttime= minimum(gdf.time),
    firstphase = minimum(gdf.phase),
    count = length(gdf.time),
    wblen = mean(gdf.wblen),
    tz = mean(gdf.tz),
    stim = mean(gdf.pos)
    )
end 
d.stim = (d.stim .- mean(d.stim)) * pixel_conversion

num = length(unique(d.muscle))
ct = select(d,:wb,:muscle,:count,:trial,:tz)
countmat = Matrix(dropmissing((select(unstack(ct,:muscle,:count),Not(:wb)))))
time = select(d,:wb,:muscle,:firstphase,:tz,:trial)
timemat = Matrix(dropmissing(select(unstack(time,:muscle,:firstphase),Not(:wb))))
tris = Bool.(countmat[:,1])
stim = countmat[:,2]

# for i in 3:12
#     timemat[:,i] .-= mean(timemat[:,i])

# end
## 
F = Figure()
ax = Axis(F[1,1])
for i in 2:11 
    scatter!(ax,timemat[:,i],timemat[:,1],label=muscle_names[i-1])
end
axislegend(ax)
F
##
m = fit(MultivariateStats.PCA,timemat[:,3:end][timemat[:,2].==0,:]')
lat = m.proj' * timemat[:,3:end]'

f = Figure() 
ax = Axis3(f[1,1])
scatter!(ax,lat[1,:],lat[2,:],lat[3,:],color=timemat[timemat[:,2].==0,1])
f 
##
pairs = collect(combinations(1:10, 3))  # 45 pairs
dat = timemat[timemat[:,2].==0,3:end]

pairwise_diffs = hcat([map(r -> norm([dat[r, i], dat[r, j], dat[r, k]]), 1:size(dat,1)) for (i,j,k) in pairs]...)

m = fit(MultivariateStats.PCA,pairwise_diffs')

lat = m.proj' * pairwise_diffs'
a,co
f = Figure() 
ax = Axis(f[1,1])
lines!(ax,lat[3,:],color=timemat[timemat[:,2].==0,1])
f 
##
using Distances
dat = timemat[:,3:end]
yaw = timemat[:,1]
ax = dat[:,1] - dat[:,10]
ba = dat[:,2] - dat[:,9]
sa = dat[:,3] - dat[:,8]
dvm = dat[:,4] - dat[:,7]
dlm = dat[:,5] - dat[:,8]

f = Figure() 
ax1 = Axis(f[1,1],title="AX")
scatter!(ax1,ax,yaw)
ax2= Axis(f[1,2],title="BA",limits=(nothing,0.1,nothing,nothing))
scatter!(ax2,ba,yaw)
ax3 = Axis(f[2,1],title="SA",limits=(nothing,0.3,nothing,nothing))
scatter!(ax3,sa,yaw)
ax4 = Axis(f[2,2],title="DVM",limits=(nothing,0.2,nothing,nothing))
scatter!(ax4,dvm,yaw)
ax5= Axis(f[1:2,3],title="DLM")
scatter!(ax5,dlm,yaw)
f
##
bilat_diffs = hcat(ax,ba,sa,dvm,dlm)
m_bilat = fit(MultivariateStats.PCA,bilat_diffs')
lat = m_bilat.proj' * bilat_diffs'
##
F = Figure() 
ax1 = Axis(F[1,1])
scatter!(ax1,lat[1,:],lat[2,:],color=yaw)
F