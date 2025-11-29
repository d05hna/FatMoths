using JSON, DataFrames, FFTW, GLMakie,AlgebraOfGraphics,StatsBase,Pipe,DataFramesMeta
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)
include("me_functions.jl")
fs = Int(1e4) 
N = Int(1e5)
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
##
d = JSON.parsefile("/home/doshna/Downloads/latents.json")

flower = d["flower"]
flower = reduce(hcat,flower)' |> x -> Float64.(x) 
##
moth = d["moth_6"]
low_share = reduce(hcat,moth["low_share_latent"])' |> x -> Float64.(x)
low = low_share[:,1:4]
share_low = low_share[:,5:end]
high_share = reduce(hcat,moth["share_high_latent"])' |> x -> Float64.(x)
share_high = high_share[:,1:4]
high = high_share[:,5:end]
##
""" 
Lets First See if there is a gain Change in the Shared Latents
"""

# Raw FFTs 
fsl = fft(share_low) 
fsh = fft(share_high)

fr = fftfreq(N,fs)

f = Figure(size=(1000,1000))
ax1 = Axis(f[1,1],title = "Shared 1",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(fsl)[2:200,1] ./ N,color=:steelblue,linewidth=3)
lines!(ax1,fr[2:200],abs.(fsh)[2:200,1] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[1,2],title = "Shared 2",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(fsl)[2:200,2] ./ N,color=:steelblue,linewidth=3)
lines!(ax1,fr[2:200],abs.(fsh)[2:200,2] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,1],title = "Shared 3",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(fsl)[2:200,3] ./ N,color=:steelblue,linewidth=3)
lines!(ax1,fr[2:200],abs.(fsh)[2:200,3] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,2],title = "Shared 4",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
l= lines!(ax1,fr[2:200],abs.(fsl)[2:200,4] ./ N,color=:steelblue,linewidth=3)
h = lines!(ax1,fr[2:200],abs.(fsh)[2:200,4] ./ N,color=:firebrick,linewidth=3)
g = vlines!(ax1,freqqs,color=:grey,alpha=0.5)

Legend(f[3,:],[l,h,g],["Low Mass","High Mass","Driving Frequencies"],orientation=:horizontal)
save("Figs/latents/Shared_raw.png",f,px_per_unit=4)
f
## Transfer Functions 


h_low = zeros(18,4) |> x -> Complex.(x)
h_high = zeros(18,4) |> x -> Complex.(x)

for i in 1:4 
    h_low[:,i] = tf_freq(flower[:,1],share_low[:,i],freqqs,fs)
    h_high[:,i] = tf_freq(flower[:,1],share_high[:,i],freqqs,fs)
end

f = Figure(size=(1000,1000))
ax1 = Axis(f[1,1],title = "Shared 1",xscale=log10,)
lines!(ax1,freqqs,abs.(h_low[:,1]),color=:steelblue,linewidth=3)
lines!(ax1,freqqs,abs.(h_high[:,1]),color=:firebrick,linewidth=3)

ax1 = Axis(f[1,2],title = "Shared 2",xscale=log10)
lines!(ax1,freqqs,abs.(h_low[:,2]),color=:steelblue,linewidth=3)
lines!(ax1,freqqs,abs.(h_high[:,2]),color=:firebrick,linewidth=3)

ax1 = Axis(f[2,1],title = "Shared 3",xscale=log10)
lines!(ax1,freqqs,abs.(h_low[:,3]),color=:steelblue,linewidth=3)
lines!(ax1,freqqs,abs.(h_high[:,3]),color=:firebrick,linewidth=3)

ax1 = Axis(f[2,2],title = "Shared 4",xscale=log10)
l= lines!(ax1,freqqs,abs.(h_low[:,4]),color=:steelblue,linewidth=3)
h = lines!(ax1,freqqs,abs.(h_high[:,4]),color=:firebrick,linewidth=3)

Legend(f[3,:],[l,h],["Low Mass","High Mass"],orientation=:horizontal)
save("Figs/latents/Shared_tf.png",f,px_per_unit=4)
f

##  high and low specific Raw 

lowft = fft(low)
f = Figure(size=(1000,1000))
ax1 = Axis(f[1,1],title = "Low 1",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,1] ./ N,color=:steelblue,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[1,2],title = "Low 2",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,2] ./ N,color=:steelblue,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,1],title = "Low 3",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,3] ./ N,color=:steelblue,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,2],title = "Low 4",xscale=log10,limits=(nothing,nothing,0,0.6),xticks=[0.1,1,10])
l= lines!(ax1,fr[2:200],abs.(lowft)[2:200,4] ./ N,color=:steelblue,linewidth=3)
g = vlines!(ax1,freqqs,color=:grey,alpha=0.5)

Legend(f[3,:],[l,g],["Low Mass","Driving Frequencies"],orientation=:horizontal)
save("Figs/latents/low_raw.png",f,px_per_unit=4)
f
##
lowft = fft(high)
f = Figure(size=(1000,1000))
ax1 = Axis(f[1,1],title = "High 1",xscale=log10,limits=(nothing,nothing,0,0.8),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,1] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[1,2],title = "High 2",xscale=log10,limits=(nothing,nothing,0,0.8),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,2] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,1],title = "High 3",xscale=log10,limits=(nothing,nothing,0,0.8),xticks=[0.1,1,10])
lines!(ax1,fr[2:200],abs.(lowft)[2:200,3] ./ N,color=:firebrick,linewidth=3)
vlines!(ax1,freqqs,color=:grey,alpha=0.5)

ax1 = Axis(f[2,2],title = "High 4",xscale=log10,limits=(nothing,nothing,0,0.8),xticks=[0.1,1,10])
l= lines!(ax1,fr[2:200],abs.(lowft)[2:200,4] ./ N,color=:firebrick,linewidth=3)
g = vlines!(ax1,freqqs,color=:grey,alpha=0.5)

Legend(f[3,:],[l,g],["High Mass","Driving Frequencies"],orientation=:horizontal)
save("Figs/latents/high_raw.png",f,px_per_unit=4)
f
##
""" 
Coherence With stimulus 
""" 
m = "moth_0"
flower = d["flower"]
flower = reduce(hcat,flower)' |> x -> Float64.(x) 


##
fr = round.(fftfreq(N,fs),digits=4)
stimidx= [i for i in 1:N if fr[i] in freqqs]

cohs_low = zeros(18,4,10)
cohs_high = zeros(18,4,10)
cohs_shared_low = zeros(18,4,10)
cohs_shared_high = zeros(18,4,10)


for i in 0:9 
    moth = d["moth_$i"]
    low_share = reduce(hcat,moth["low_share_latent"])' |> x -> Float64.(x)
    low = low_share[:,1:4]
    share_low = low_share[:,5:end]
    high_share = reduce(hcat,moth["share_high_latent"])' |> x -> Float64.(x)
    share_high = high_share[:,1:4]
    high = high_share[:,5:end]
    for j in 1:4
        l = mt_coherence(hcat(low[:,j],flower)';fs=fs,nfft=N).coherence[1,2,stimidx]
        h = mt_coherence(hcat(high[:,j],flower)';fs=fs,nfft=N).coherence[1,2,stimidx]
        sl = mt_coherence(hcat(share_low[:,j],flower)';fs=fs,nfft=N).coherence[1,2,stimidx]
        sh = mt_coherence(hcat(share_high[:,j],flower)';fs=fs,nfft=N).coherence[1,2,stimidx]

        cohs_low[:,j,i+1] = l 
        cohs_high[:,j,i+1] = h 
        cohs_shared_low[:,j,i+1] = sl 
        cohs_shared_high[:,j,i+1] = sh 
    end 
end 
##
f = Figure(size=(800,800)) 
ax = Axis(f[1,1],xticks = ([1,2,3,4],["L1","L2","L3","L4"]),yticks = (range(1,18,length=18),string.(freqqs)),
    title= "Mean Coherence Low Specific")
h = heatmap!(ax,mean(cohs_low,dims=3)[:,:,1]',colorrange=(0.4,1),colormap=:viridis)
ax = Axis(f[1,2],xticks = ([1,2,3,4],["H1","H2","H3","H4"]),yticks = (range(1,18,length=18),string.(freqqs)),
    title= "Mean Coherence High Specific")
h = heatmap!(ax,mean(cohs_high,dims=3)[:,:,1]',colorrange=(0.4,1),colormap=:viridis)
ax = Axis(f[2,2],xticks = ([1,2,3,4],["S1","S2","S3","S4"]),yticks = (range(1,18,length=18),string.(freqqs)),
    title= "Mean Coherence High Shared")
h = heatmap!(ax,mean(cohs_shared_high,dims=3)[:,:,1]',colorrange=(0.4,1),colormap=:viridis)
ax = Axis(f[2,1],xticks = ([1,2,3,4],["S1","S2","S3","S4"]),yticks = (range(1,18,length=18),string.(freqqs)),
    title= "Mean Coherence Low Shared")
h = heatmap!(ax,mean(cohs_shared_low,dims=3)[:,:,1]',colorrange=(0.4,1),colormap=:viridis)
Colorbar(f[:,3],h)
save("Figs/latents/Mean_Coherence.png",f,px_per_unit=3)
f
## 
freq_idx = repeat(freqqs,inner=40)
trial_idx = repeat(repeat(1:4, inner=10), outer=18)
participant_idx = repeat(1:10, outer=18*4)
values = vec(cohs_shared_high)
df_low = DataFrame(
    freq = freq_idx,
    latent = ["Shared (High) $i" for i in trial_idx],
    moth = participant_idx,
    Coherence = values
)

logx = log10.(freqqs[1:18])
Δ = 0.06 # width in log units
left  = 10 .^ (logx .- Δ/2)
right = 10 .^ (logx .+ Δ/2)
widths = right .- left

df_low.widths = repeat(widths,inner=40)
plot = data(df_low)*mapping(:freq,:Coherence,layout=:latent) *visual(color=:firebrick)
f = draw(plot,axis=(; xscale=log10,xticks=[0.2,1,10]))