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

function process_data(file, fs,starting_point,bias)
    ftnames = ["fx", "fy", "fz", "tx", "ty", "tz"]

    df = h5_to_df(file)
    true_ft = DataFrame(transform_FT(transpose(Matrix(df[!, ftnames]).-bias)), ftnames)
    df[!, ftnames] = true_ft
    # idx = findall(df.camtrig .> 2)[1]
    # println(idx)
    # start = Int(idx+fs*starting_point)+1
    start = Int(fs*starting_point)+1
    # start = idx
    fx = df.fx[start:(start+10*fs)-1]
    # fx = df.fx[start:end]
    full = df[start:((starting_point +10)*fs),ftnames]
    return full
end
function process_stim(stim_file)
    stim = CSV.read(stim_file, DataFrame)
    stim_20 = convert(Vector{Float64}, stim.filtered[1:6000])
    return stim_20
end
function calculate_fft(data, fs)
    fft_data = fft(data)[2:end]
    fr_data = fftfreq(length(data), fs)[2:end]
    return fr_data, fft_data
end
function get_raw_fig(moth,datadir,timingdic)
    fs = 10000
    startpre = timingdic[moth]["pre"][1]
    startpost = timingdic[moth]["post"][1]
    files = glob("$(moth)/*.h5", datadir)

    pre = files[1]
    post = files[3]

    fxpre = process_data(pre,fs,startpre)
    fxpost = process_data(post,fs,startpost)

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
    # ax.limits=(0,20,nothing,nothing)r
    axislegend(ax, "Trial", position=:rt)
    return(fig)
end

function get_fft_fig(moth,datadir,timingdic)
    files = glob("$(moth)/*.h5", datadir)
    fs = 10000

    startpre = timingdic[moth]["pre"][1]
    startpost = timingdic[moth]["post"][1]

    pre = files[1]
    post = files[3]

    fxpre = process_data(pre,fs,startpre)
    fxpost = process_data(post,fs,startpost)


    freqpre, fftpre = calculate_fft(fxpre,fs)
    freqpost,fftpost = calculate_fft(fxpost,fs)

    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]

    idxs = freqpre .>0 .&& freqpre .<10

    fridx = findall(x->x in freqqs,round.(freqpre,digits=2))

    peakspre = abs.(fftpre[fridx])
    peakspost = abs.(fftpost[fridx])

    fig = Figure(size=(800, 600), fontsize=14)
    ax = Axis(fig[1, 1], 
        xlabel = "Frequency", 
        ylabel = "Amplitude",
        title = "$moth Frequency Response",
        xscale = log10
    )
    lines!(ax,freqpre[idxs],abs.(fftpre[idxs]),label="Before Feeding",alpha=0.5)
    lines!(ax,freqpost[idxs],abs.(fftpost[idxs]),label="After Feeding",alpha=0.5)
    axislegend(ax, "Trial", position=:rt)
    vlines!(ax,freqqs,color=:grey,linestyle=:dash,alpha=0.3)




    return(fig,peakspre,peakspost)
end

function get_gain(moth,datadir,freqqs,timingdic)
    files = glob("$(moth)/*.h5", datadir)
    stims = glob("$(moth)/DLT_data/munged*",datadir) 

    startpre = timingdic[moth]["pre"][1]
    startpost = timingdic[moth]["post"][1]

    fs = 10000


    pre = files[1]
    post = files[3]

    ##
    fxpre = process_data(pre,fs,startpre)
    fxpost = process_data(post,fs,startpost)

    stimpre = process_stim(stims[1])
    stimpost = process_stim(stims[3])

    ##



    fx_fr = fftfreq(length(fxpre),fs)

    idx = findall(x->x in freqqs,round.(fx_fr,digits=2))


    fft_pre_fx = fft(fxpre)[idx]
    fft_post_fx = fft(fxpost)[idx]

    fft_stim_pre = fft(stimpre)[idx]
    fft_stim_post = fft(stimpost)[idx]

    ##
    gain_pre = fft_pre_fx ./ fft_stim_pre 
    gain_post = fft_post_fx ./ fft_stim_post

    return(abs.(gain_pre),abs.(gain_post))
end

##
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")
moths = ["2024_08_01","2024_08_12","2024_08_14","2024_08_15","2024_08_16"]
# Lets Just look at the raw and the peaks 
timingdic = Dict(
    "2024_08_01"=> Dict("pre"=>[0,20], "post"=>[0,20]),
    "2024_08_12" => Dict("pre"=>[2,12], "post"=>[11,21 ]),
    "2024_08_14" => Dict("pre"=>[16,26], "post"=>[15,25]),
    "2024_08_15" => Dict("pre"=>[7,17], "post"=>[9,19]),
    "2024_08_16" => Dict("pre"=>[12,22], "post"=>[10,20])
)

##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]
allpeaks = DataFrame()

for moth in moths
    f = get_raw_fig(moth,datadir,timingdic)
    save("newtracking/POSITION$(moth).png",f,px_per_unit=4)
    g,ppre,ppost = get_fft_fig(moth,datadir,timingdic)
    save("newtracking/FFT$(moth).png",g,px_per_unit=4)
    tmp = DataFrame(
        freq = freqqs,
        pre = ppre,
        post = ppost,
        change = ppost ./ ppre
    )
    tmp.moth .= moth
    allpeaks = vcat(allpeaks,tmp,cols=:union)

end
##
for moth in moths
    d4t = allpeaks[allpeaks.moth.==moth,:]
    fig = Figure(size=(800, 600), fontsize=14)
    ax = Axis(fig[1, 1], 
        xlabel = "Frequency", 
        ylabel = "% Change in FFT Peak",
        title = "$moth Change in FFT Peak",
        xscale = log10
    )
    scatter!(ax,d4t.freq,d4t.change,color=:red)
    save("newtracking/CHANGE$(moth).png",fig,px_per_unit=4)
end
##

"""
Now that we have all the raw stuff processed lets look at the gain using The method simon said

The FX/Flower in the complex plane is N 

NM is N * 1/{mass(2pifi)^2) 

we need NM/1+NM for the closed loop response? 

But masses are broken before we go back to this 

We can just look at the Trasnfer function in open loop, by looking at the N 
Take flower/moth fx  in complex plane, average and get iqr in complex plane 
"""



##

freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]

for moth in moths
    gain_pre,gain_post = get_gain(moth,datadir,freqqs,timingdic)

    fig = Figure()
    ax = Axis(fig[1,1],xscale=log10,xlabel="Freq",ylabel="abs of fx/flower",title = "Gain of $moth",
        limits= (nothing,10,nothing,nothing))
    scatter!(ax,freqqs,gain_pre,label="pre")
    scatter!(ax,freqqs,gain_post,label="post")
    axislegend(ax, "Trial", position=:lt)
    
    save("newtracking/GAIN$(moth).png",fig,px_per_unit=4)
end
##
