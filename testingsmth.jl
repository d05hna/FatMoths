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
using Distributions
using Interpolations
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
# theme.palette = (color = [:turquoise,:coral],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
@load "fat_moths_set_1.jld" allmoths
fs = 1e4 
##
function get_gaus_conv(signal,sig_time=0.01) #sig time is in ms 
    sig = sig_time * (fs/1e3) # samples

    kernel_width = ceil(Int,3sig)
    kernel_t = (-kernel_width:1/fs:kernel_width)
    gaus_ker = pdf.(Normal(0,sig_time),kernel_t)
    gaus_ker ./ sum(gaus_ker) #normalize 

    res = conv(signal,gaus_ker)
    offset = (length(res) - length(signal)) ÷ 2

    res = res[offset+1:end-offset]
    return(res)
end
##
function check_coh(allmoths,moth,muscle1,muscle2)
    fpre = allmoths[moth]["fxpre"]
    fpost = allmoths[moth]["fxpost"]
    df = allmoths[moth]["data"]
    dfpre = df[df.trial.=="pre",:]
    dfpre.time_abs = round.(dfpre.time_abs .- minimum(dfpre.time_abs),digits=4)
    dfpost = df[df.trial.=="post",:]
    dfpost.time_abs = round.(dfpost.time_abs .- minimum(dfpost.time_abs),digits=4)
    ##

    rdlm_pre = dfpre[dfpre.muscle.==muscle1,:time_abs]
    ldlm_pre = dfpre[dfpre.muscle.==muscle2,:time_abs]

    delt_r_pre = zeros(Int(1e5))
    delt_r_pre[Int.(round.(rdlm_pre*fs,digits=0)).+1] .= 1

    delt_l_pre = zeros(Int(1e5))
    delt_l_pre[Int.(round.(ldlm_pre*fs,digits=0)).+1] .= 1

    rdlm_post = dfpost[dfpost.muscle.==muscle1,:time_abs]
    ldlm_post = dfpost[dfpost.muscle.==muscle2,:time_abs]

    delt_r_post = zeros(Int(1e5))
    delt_r_post[Int.(round.(rdlm_post*fs,digits=0)).+1] .= 1

    delt_l_post= zeros(Int(1e5))
    delt_l_post[Int.(round.(ldlm_post*fs,digits=0)).+1] .= 1

    cont_r_pre = get_gaus_conv(delt_r_pre)
    cont_l_pre = get_gaus_conv(delt_l_pre)

    lr_diff_cont = cont_l_pre - cont_r_pre


    cont_r_post = get_gaus_conv(delt_r_post)
    cont_l_post = get_gaus_conv(delt_l_post)

    lr_diff_cont_post = cont_l_post - cont_r_post

    catpost = vcat(fpost',lr_diff_cont_post')
    catpre = vcat(fpre',lr_diff_cont')


    cohpost = mt_coherence(catpost; fs=fs,nw=2,nfft=length(catpost[1,:]))
    cohpre = mt_coherence(catpre; fs=fs,nw=2,nfft=length(catpre[1,:]))

    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

    time = 0+1/fs:1/fs:10

    fig = Figure(resolution=(1000,1000))
    fig[1,:] = Label(fig, "Offset Coherence - $muscle1 $muscle2", fontsize=20)

    ax1 = Axis(fig[2,1],xlabel="time",ylabel="muscle time offset",title="pre")
    lines!(ax1,time,lr_diff_cont,color=:turquoise)
    ax2 = Axis(fig[2,2],xlabel="time",ylabel="Fx",title="pre")
    lines!(ax2,time,fpre,color=:turquoise)
    ax3 = Axis(fig[3,1],xlabel="time",ylabel="muscle time offset",title="post")
    lines!(ax3,time,lr_diff_cont_post,color=:coral)
    ax4 = Axis(fig[3,2],xlabel="time",ylabel="Fx",title="post")
    lines!(ax4,time,fpost,color=:coral)

    ax5 = Axis(fig[4,1],xlabel="freq",ylabel="coh",title="pre",xscale=log10,limits=(nothing,20,nothing,nothing))
    lines!(ax5,cohpre.freq,cohpre.coherence[2,1,:],color=:turquoise)
    vlines!(ax5,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

    ax6 = Axis(fig[4,2],xlabel="freq",ylabel="coh",title="post",xscale=log10,limits=(nothing,20,nothing,nothing))
    lines!(ax6,cohpost.freq,cohpost.coherence[2,1,:],color=:coral)
    vlines!(ax6,freqqs,color=:grey,linestyle=:dash,alpha=0.3)



    return(lr_diff_cont,lr_diff_cont_post,cohpre,cohpost,fig)
end

##
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
m = "2024_11_20"
df = allmoths[m]["data"]
# full = get_big_data(df,"time")

##
function analyze_muscle_timing(df::DataFrame,m1,m2)
    # Group by wb
    results = combine(groupby(df, :wb)) do group_df
        # Get minimum times for each muscle type
        ldlm_rows = filter(row -> row.muscle == m1, group_df)
        rdlm_rows = filter(row -> row.muscle == m2, group_df)
        
        # Calculate min time difference
        time_diff = if !isempty(ldlm_rows) && !isempty(rdlm_rows)
            minimum(ldlm_rows.time) - minimum(rdlm_rows.time)
        else
            missing
        end
        
        # Calculate average time_abs
        avg_time_abs = mean(group_df.time_abs)
        
        # Return named tuple with results
        (
            time_difference = time_diff,
            average_time_abs = avg_time_abs
        )
    end
    leftjoin!(results,unique(select(df,:wb,:trial)),on=:wb)
    return results
end

function get_df_dlms(moth,allmoths)
    df = allmoths[moth]["data"]
    wdf = select(df,:wb,:time_abs,:muscle,:time,:trial)
    dlms = wdf[wdf.muscle .=="ldlm" .|| wdf.muscle .== "rdlm",:]

    diffs = analyze_muscle_timing(dlms,"ldlm","rdlm")
    dropmissing!(diffs)
    pred = diffs[diffs.trial.=="pre",:]

    # pred = pred[pred.time_difference.>-0.005 .&& pred.time_difference .< 0.005,:]
    posd = diffs[diffs.trial.=="post",:]

    pred.wb .-= minimum(pred.wb)
    posd.wb .-= minimum(posd.wb)

    pred.score = zscore(pred.time_difference)
    posd.score = zscore(posd.time_difference)

    fu = vcat(pred,posd,cols=:union)
    select!(fu,:trial,:wb,:time_difference,:score)
    fu.moth .= moth
    return(fu)
end
##
moths = collect(keys(allmoths))
bad_moths = ["2024_08_01","2024_06_06","2024_06_20","2024_06_24","2024_12_04_2"]
moths = [m for m in moths if !in(m,bad_moths)]
##
all_dlms = DataFrame()
for moth in moths
    d = get_df_dlms(moth,allmoths)
    all_dlms = vcat(all_dlms,d,cols=:union)
end


##
ord = 2
cutoff = 5
responsetype= Lowpass(cutoff;fs=fs)
met = Butterworth(ord)

lpfilt = digitalfilter(responsetype,met)
##

wdf = select(df,:wb,:time_abs,:muscle,:time,:trial)
dlms = wdf[wdf.muscle .=="ldlm" .|| wdf.muscle .== "rdlm",:]

diffs = analyze_muscle_timing(dlms,"ldlm","rdlm")
dropmissing!(diffs)
pred = diffs[diffs.trial.=="pre",:]

# pred = pred[pred.time_difference.>-0.005 .&& pred.time_difference .< 0.005,:]
posd = diffs[diffs.trial.=="post",:]
# posd = posd[posd.time_difference .> -0.002 .&& posd.time_difference .< 0.002,:]


##

##
l = CubicSplineInterpolation(range(0,10,length=nrow(pred)),pred.time_difference)
s = 0 
en = last(l.itp.ranges[1])

new_time_vec = range(s,en,length=Int(10*fs))

pred_long = l.itp.(new_time_vec)

pred_long .-= mean(pred_long)
predf =filtfilt(lpfilt,pred_long)


##
w = CubicSplineInterpolation(range(0,10,length=nrow(posd)),posd.time_difference)
s = 0 
en = last(w.itp.ranges[1])

new_time_vec = range(s,en,length=Int(10*fs))

posd_long = w.itp.(new_time_vec)
posd_long .-= mean(posd_long)

posdf = filtfilt(lpfilt,posd_long)
##
# freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
using GLMakie
GLMakie.activate!()
sicbTheme = theme_minimal()
sicbTheme.textcolor=:black
sicbTheme.ytickfontcolor=:black
sicbTheme.fontsize= 30
sicbTheme.gridcolor=:lightcyan
sicbTheme.gridalpga=0.5
# sicbTheme.palette = (color = [:turquoise,:coral],)
 
set_theme!(sicbTheme)
##
freqqs = [0.200, 0.300, 0.500, 0.700]


fftpre = abs.(fft(predf))[2:20]
freqrange = fftfreq(Int(10*fs),fs)[2:20]

fftpost = abs.(fft(posdf))[2:20]

fig = Figure(resolution=(1200,600))
ax = Axis(fig[1,1],xscale=log10,ylabel="Amplitude",xlabel="Frequency")
hi = lines!(ax,freqrange,fftpre,color=:firebrick)
lo = lines!(ax,freqrange,fftpost,color=:deepskyblue)
dr = vlines!(ax,freqqs,color=:grey,linewidth=3,alpha=0.3)
Legend(fig[1,2],[lo,hi,dr],["Low Mass","High Mass","Flower Frequency"])
# save("SICBFigs/DLMGAIN.png",fig,px_per_unit=4)
fig


##
t = 1/fs:1/fs:10
f = Figure(resolution=(1200,600))
ax = Axis(f[1,1],xlabel="Time (s)",ylabel = "DLM Offset (ms)")
lo = lines!(ax,t,posdf*1e3,color=:deepskyblue,linewidth=5)
hi = lines!(ax,t,predf*1e3,color=:firebrick,linewidth=5)
Legend(f[1,2],[lo,hi],["Low Mass","High Mass"])
# save("SICBFigs/DLMTIME.png",f,px_per_unit=4)

f
##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30]
# 

fpre = allmoths[m]["fxpre"]
fpost = allmoths[m]["fxpost"]

freqrange = fftfreq(Int(10*fs),fs)[2:100]

ftpref = abs.(fft(fpre))[2:100]
ftposf = abs.(fft(fpost))[2:100]



fig = Figure(resolution=(1200,600))


ax2 = Axis(fig[1,1],xscale=log10,xlabel="Frequency",ylabel="Amplitude")
r = lines!(ax2,freqrange,ftposf,color=:steelblue,linewidth=4)
l = vlines!(ax2,freqqs,color=:grey,linewidth=3,alpha=0.3)
Legend(fig[1,2],[r,l],["Moths Response","Flower Frequency"])
save("SICBFigs/respfft.png",fig,px_per_unit=4)

fig