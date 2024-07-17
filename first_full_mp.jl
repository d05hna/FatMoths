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
##
"""
It fed really slowly, and the beggining took a bit. I am going to compare the last 20 
seconds of the first trial (when I first started to get good flapping) and the last 20 
seconds I recorded. It is not perfect, but I can't even find out the gains rn and the 
tracking was pretty shit anyways 
"""

p1 = CSV.read("pre_moth_full.csv",DataFrame)
p2 = CSV.read("post_moth_full.csv",DataFrame)
##
p1.wb = p1.wb .- minimum(p1.wb)
p2.wb = (p2.wb .- minimum(p2.wb)) .+ maximum(p1.wb)
df=vcat(p1,p2)


## Lets look at WB freq

d4t = select(df,:wb,:trial,:wblen)
d4t.freq = 1 ./ d4t.wblen

plot = data(d4t)*mapping(:freq,color=:trial => "Epoch")*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
f = draw(plot,axis=(; title="Wb Frequency at the Start and End of Feeding",xlabel="Wing Beat Frequency",ylabel=""))
# save("wbfreq.png",f,px_per_unit=4)
f
## Lets look at all the muscle phases
plot = data(df)*mapping(:phase,color=:trial=>"Epoch",row=:muscle)*histogram(bins=100,normalization=:probability)*visual(alpha=0.5)
fig = draw(plot,axis=(;))
save("musclephases.png",fig,px_per_unit=4)
fig