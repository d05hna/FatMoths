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
using ProgressMeter
using Statistics 
using StatsBase
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("/home/doshna/Documents/PHD/comparativeMPanalysis/readAndPreprocessFunctions.jl")
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
data_dir = joinpath("/home", "doshna", "Documents", "PHD", "data", "fatties")
# Controls
fs = 10000
emg_highpass = 70
ft_lowpass = 1000
z_bandpass = [10, 50]
wb_len_thresh = Dict(
                    "Manduca sexta" => [1/30, 1/15]) 
phase_wrap_thresh = Dict(
    "Manduca sexta"  => Dict(
        "2024_11_01" => Dict("ax"=>2.0, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_04" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_05" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_07" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_08" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_11" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2024_11_20" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2025_01_14" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2025_01_23" => Dict("ax"=>0.9, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7),
        "2025_01_30" => Dict("ax"=>2.0, "ba"=>2.0, "sa"=>0.7, "dvm"=>0.41, "dlm"=>0.7)
        )
)
cheby_bandpass = digitalfilter(Bandpass(z_bandpass...; fs=fs), Chebyshev1(4, 4))

muscle_names = ["lax","lba","lsa","ldvm","ldlm",
"rdlm","rdvm","rsa","rba","rax"]
ft_names = ["fx","fy","fz","tx","ty","tz"]

##
moth = "2025_01_30"

##
gtp = Dict(
    ## Moth => Trial 1, Trial 2, Start 1, Start 2, Trials are 1 indexed
    "2024_11_01" => [2,4,1.5e5,1],
    "2024_11_04" => [2,4,1.5e5,1],
    "2024_11_05" => [1,4,1,1e5],
    "2024_11_07" => [1,3,1e4,1],
    "2024_11_08" => [2,4,1e5,1],
    "2024_11_11" => [1,2,1,1],
    "2024_11_20" => [1,3,1e5,1.5e5],
    "2025_01_23" => [1,3,1e5,2e5],
    "2025_01_30" => [1,2,1e4,1.25e5]


)

##
df = DataFrame()
params = Dict()
df_ft_all = DataFrame()
read_individual!(joinpath(data_dir,moth),df,df_ft_all,params,wb_len_thresh,phase_wrap_thresh;cheby_bandpass=cheby_bandpass)
df.time_abs .+= 30
fx_pre,fx_post = get_side_slips(data_dir,moth,params[moth],[Int(gtp[moth][1]),Int(gtp[moth][2])])
##
"""
STOP RIGHT HERE!!! LOOK Between Videos and Force Traces and Get the Best 10 Seconds of Tracking for each!!
Then Adjust the Following Variables accordingly
Also if the pre and post trials are not 0 and 2, then change those too!!
"""
## Filter the Fx data and the Muscle Data To only care about the Portions that are being used in the tracking analysis
start_pre = Int(gtp[moth][3])
start_post = Int(gtp[moth][4])

pre10 = fx_pre[start_pre:Int(start_pre+1e5-1)]
post10 = fx_post[start_post:Int(start_post+1e5-1)]

tris = convert(Vector{Int},[gtp[moth][1],gtp[moth][2]] .- 1)
predf = df[df.trial.==tris[1],:]
predf = predf[predf.time_abs .> start_pre /fs .&& predf.time_abs .< (start_pre-1+1e5)/fs,:]
predf.trial .= "pre"

postdf = df[df.trial.==tris[2],:]
postdf = postdf[postdf.time_abs .> start_post/fs .&& postdf.time_abs .< (start_post-1+1e5)/fs,:]
postdf.trial .= "post"

df_to_use = vcat(predf,postdf,cols=:union)

## peak at the wb freq
plot = data(df_to_use)*mapping(:wbfreq,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) |> draw
## Take a look at mean z forces
plot = data(df_to_use)*mapping(:fz,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) |> draw
## Take a look at Muscle Activities
plot = data(df_to_use) * mapping(:phase,row=:muscle,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=250,normalization=:probability)*visual(alpha=0.5) |> draw
## Take a Look at the frequency Responses 
fftpre = abs.(fft(pre10)[2:50000])
fftpost = abs.(fft(post10)[2:50000])
freqrange = fftfreq(length(pre10),fs)[2:50000]
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

id = findfirst(x->x ==15, freqrange)
fftpre = fftpre[1:id]
fftpost=fftpost[1:id]
freqrange=freqrange[1:id]

fig = Figure()
ax1 = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Pre",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)
    )
ax2 = Axis(fig[1, 2], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Post",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)
    )
ax3 = Axis(fig[2, :], 
    xlabel = "Frequency", 
    ylabel = "Amplitude",
    title = "Both",
    xscale = log10,
    limits=(nothing,12,nothing,nothing)

    )


pre = lines!(ax1, freqrange,fftpre,color=:turquoise)
post = lines!(ax2, freqrange,fftpost,color=:coral)

lines!(ax3, freqrange,fftpre,color=:turquoise)
lines!(ax3, freqrange,fftpost,color=:coral)

vlines!(ax1,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
vlines!(ax2,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
vlines!(ax3,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

Legend(fig[:,:3],[pre,post],["Before Feeding","After Feeding"])
fig
## save away
@load "/home/doshna/Documents/PHD/FatMoths/fat_moths_set_1.jld" allmoths
if moth in collect(keys(allmoths))
    allmoths[moth]["data"] = df_to_use
    allmoths[moth]["fxpre"] = pre10
    allmoths[moth]["fxpost"] = post10
else
    d = Dict(
        "data" => df_to_use,
        "fxpre" => pre10,
        "fxpost" => post10
    )

    allmoths[moth] = d 
end
@save "/home/doshna/Documents/PHD/FatMoths/fat_moths_set_1.jld" allmoths

