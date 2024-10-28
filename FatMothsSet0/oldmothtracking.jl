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
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:turquoise,:coral],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
function calculate_fft(data, fs)
    fft_data = fft(data)[2:end]
    fr_data = fftfreq(length(data), fs)[2:end]
    return fr_data, fft_data
end
##
fs = 10000
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")
moth = "2024_07_09"

startimes = Dict(
    "2024_06_06" => [0,50],
    "2024_06_20" => [0,17],
    "2024_06_24" => [0,16],
    "2024_07_09"  => [3,15] # use 001 for post
)

##
files = glob("$(moth)/*.h5", datadir)
file = files[1]

##

ftnames = ["fx", "fy", "fz", "tx", "ty", "tz"]

df = h5_to_df(file)
true_ft = DataFrame(transform_FT(transpose(Matrix(df[!, ftnames]))), ftnames)
df[!, ftnames] = true_ft

select!(df,ftnames,"time")
s = startimes[moth][1]
e = startimes[moth][2]
fxpre = df[Int(s*fs)+1:Int((s+10)*fs),:fx]
# fxpost = df[Int(e*fs)+1:Int((e+10)*fs),:fx]
##
timepre= range(0, step=1/fs, length=length(fxpre))
timepost = range(0,step=1/fs,length=length(fxpost))
##
colors = [:turqoiuse, :coral]

fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Time (seconds)", 
    ylabel = "X Force (newtons)",
    title = "$moth FX During Tracking",
)
lines!(ax,timepre,fxpre,label="Before Feeding",alpha=0.5)
lines!(ax,timepost,fxpost,label="After Feeding",alpha=0.5)
ax.limits=(0,10,nothing,nothing)
axislegend(ax, "Trial", position=:rt)
save("newtracking/POSITION$(moth).png",fig,px_per_unit=4)
fig




##

frpre, fftpre = calculate_fft(fxpre,fs)
frpost, fftpost = calculate_fft(fxpost,fs)


freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]

idxs = frpost .>0 .&& frpost .<10

fridx = findall(x->x in freqqs,round.(frpre,digits=2))

peakspre = abs.(fftpre[fridx])
peakspost = abs.(fftpost[fridx])

fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "$moth Frequency Response",
    xscale = log10
)
lines!(ax,frpre[idxs],abs.(fftpre[idxs]),label="Before Feeding",alpha=0.5)
lines!(ax,frpost[idxs],abs.(fftpost[idxs]),label="After Feeding",alpha=0.5)
axislegend(ax, "Trial", position=:rt)
vlines!(ax,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

# save("newtracking/FFT$( moth).png",fig,px_per_unit=4)

fig

##

tmp = DataFrame(
    freq = freqqs,
    pre = peakspre,
    post = peakspost,
    change = peakspost ./ peakspre
)

tmp.moth .= moth
##

