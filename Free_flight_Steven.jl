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
##
@load "fat_moth_free_flight.jld" FreeFlight 
##

m1 = FreeFlight["moth_3"]

##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
mat_pre = zeros(length(freqqs),length(keys(FreeFlight)))
mat_post = zeros(length(freqqs),length(keys(FreeFlight)))
angle_pre = zeros(length(freqqs),length(keys(FreeFlight)))
angle_post = zeros(length(freqqs),length(keys(FreeFlight)))

for (i,moth) in enumerate(keys(FreeFlight))
    m1 = FreeFlight[moth]
    fr = round.(fftfreq(2000,100),digits=4)

    stimidx= [i for i in 1:length(m1["moth_pre"]) if fr[i] in freqqs]

    fmpr = fft(m1["moth_pre"])[1:402] 
    fmpo = fft(m1["moth_post"])[1:402] 
    ffpr = fft(m1["flower_pre"])[1:402] 
    ffpo = fft(m1["flower_post"])[1:402] 
    fr = fr[2:402]

    tf_pre = fmpr ./ ffpr
    tf_post = fmpo ./ ffpo 

    tf_pre = tf_pre ./ tf_pre[1]
    tf_post = tf_post ./ tf_post[1]
    mat_pre[:,i] = abs.(tf_pre)[stimidx]
    mat_post[:,i] = abs.(tf_post)[stimidx]

    angle_pre[:,i] = angle.(tf_pre)[stimidx]
    angle_post[:,i] = angle.(tf_post)[stimidx]

end
##

function adjusted_angle(z)
    return z <= 0 ? z : z - 2π
end
##

mean_pre = mean(mat_pre,dims=2)[:]
mean_post = mean(mat_post,dims=2)[:]

sem_pre = std(mat_pre, dims=2) ./ sqrt(size(mat_pre, 2))
sem_post = std(mat_post, dims=2) ./ sqrt(size(mat_post, 2))  

mean_pre_angle = adjusted_angle.(mean(angle_pre,dims=2)[:])
mean_post_angle = adjusted_angle.(mean(angle_post,dims=2)[:])
sem_pre_angle = std(angle_pre, dims=2) ./ sqrt(size(angle_pre, 2))
sem_post_angle = std(angle_post, dims=2) ./ sqrt(size(angle_post, 2))
##

f = Figure()
ax = Axis(f[1,1], title="Free Flight", xlabel="Frequency (Hz)", ylabel="Gain",yscale=log10,xscale=log10,yticks=[0.1,1],
    xticks = [0.1,1,10,20] ,   limits = (nothing,nothing,0.01,2))
lines!(ax,freqqs,mean_pre, color=:blue, label="Pre", linewidth=2)  
lines!(ax,freqqs,mean_post,color = :red, label="Post", linewidth=2) 
band!(ax,freqqs,(mean_pre .- sem_pre)[:], (mean_pre .+ sem_pre)[:], color = :blue, alpha=0.2)
band!(ax,freqqs,(mean_post .- sem_post)[:], (mean_post .+ sem_post)[:], color = :red, alpha=0.2)

ax2 = Axis(f[2,1],title="Phase",xlabel = "Freqeuncy (Hz)",ylabel="Phase")