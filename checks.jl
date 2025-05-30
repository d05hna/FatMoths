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
moth = "2024_11_08"
# Get the transer Function and Nyquist Plots
for moth in collect(keys(allmoths))
    pr,po = transfer_function(allmoths,moth;axis="tz")

    unpre = unwrap(angle.(pr))
    unpo = unwrap(angle.(po))

    fpre = allmoths[moth]["ftpre"][:,6]

    spre = float.(allmoths[moth]["stimpre"])

    fpo = allmoths[moth]["ftpos"][:,6]

    spo = float.(allmoths[moth]["stimpost"])

    winlen = 400
    b = ones(winlen) / winlen 
    a = [1.0]

    winlens = 12
    bs = ones(winlens)/winlens


    fpre = filtfilt(b,a,fpre)[20001:40000]
    spre = filtfilt(bs,a,spre)[601:1200]
    fpo = filtfilt(b,a,fpo)[30001:50000]
    spo = filtfilt(bs,a,spo)[901:1500]
    tf = 1/10000:1/10000:2
    ts = 1/300:1/300:2
    
    f = Figure(size=(1200,1000),title="Phase")
    ax = Axis(f[1,2],xscale=log10,ylabel="Angle (rad)",xlabel="freq",title="Phase")
    scatter!(ax,freqqs,unpre,label="pre",markersize=15)
    scatter!(ax,freqqs,unpo,label="post",markersize=15)
    axislegend(ax)

    ax2 = Axis(f[1,1],xscale=log10,ylabel="Gain (Newtons/Pixels)",xlabel="freq",title="Gain")
    scatter!(ax2,freqqs,abs.(pr),label="pre",markersize=15)
    scatter!(ax2,freqqs,abs.(po),label="post",markersize=15)
    axislegend(ax2)
    f[0, 1:2] = Label(f, "$moth", 
                    fontsize = 24, 
                    font = :bold)

    ax3 = Axis(f[2,1],title="Pre",xlabel = "Time",ylabel="Force (N")
    ax4 = Axis(f[2,1],ylabel="Flower Position",yaxisposition=:right,ylabelcolor=:forestgreen)
    lines!(ax3,tf,fpre,color=:steelblue)
    lines!(ax4,ts,spre,color=:forestgreen)

    ax5 = Axis(f[2,2],title="Post",xlabel = "Time",ylabel="Force (N")
    ax6 = Axis(f[2,2],ylabel="Flower Position",yaxisposition=:right,ylabelcolor=:forestgreen)
    lines!(ax5,tf,fpo,color=:firebrick)
    lines!(ax6,ts,spo,color=:forestgreen)

    save("checks/YawTracking/$moth.png",f)

    ## Nyq`
    fig = Figure(size=(1000,500))
    fig[0, 1:2] = Label(fig, "$moth", 
    fontsize = 24, 
    font = :bold)
    ax = Axis(fig[1,1],xlabel="Real",ylabel="Imag",title="Low Mass")
    lines!(ax,real.(pr),imag.(pr),color=:steelblue,alpha=0.5,linewidth=4)
    scatter!(ax,real.(pr),imag.(pr),color=:steelblue)

    ax2 = Axis(fig[1,2],xlabel="Real",ylabel="Imag",title="High Mass")
    lines!(ax2,real.(po),imag.(po),color=:firebrick,alpha=0.5,linewidth=4)
    scatter!(ax2,real.(po),imag.(po),color=:firebrick)

    save("checks/YawTracking/Nyquist$moth.png",fig)

end
## Just look at gain and Phase for the FX
for moth in collect(keys(allmoths))
    fxpre = allmoths[moth]["fxpre"]
    fxpost = allmoths[moth]["fxpost"]

    fr = round.(fftfreq(length(fxpre),1e4),digits=2)
    idx = [i for i in 1:length(fr) if fr[i] in freqqs]

    fftpre = fft(fxpre)[idx]
    fftpost = fft(fxpost)[idx]

    m = 15 

    f = Figure(size=(800,800))
    ax = Axis(f[1,1],xscale=log10,title="Gain",xlabel="frequency",ylabel="Magnitude of FFT (fx, n)")
    scatter!(ax,freqqs,abs.(fftpre),color=:steelblue,label="Low Mass",markersize=m)
    scatter!(ax,freqqs,abs.(fftpost),color=:firebrick,label="High Mass",markersize=m)

    ax2 = Axis(f[1,2],xscale=log10,title="Phase",xlabel="frequency",ylabel="Phase (rad) of FFT fx")
    scatter!(ax2,freqqs,angle.(fftpre),color=:steelblue,label="Low Mass",markersize=m)
    scatter!(ax2,freqqs,angle.(fftpost),color=:firebrick,label="High Mass",markersize=m)


    ax3 = Axis(f[2,2],xscale=log10,title="Phase (unwrapped)",xlabel="frequency",ylabel="Phase (rad) of FFT fx)")
    l = scatter!(ax3,freqqs,unwrap(angle.(fftpre)),color=:steelblue,label="Low Mass",markersize=m)
    h = scatter!(ax3,freqqs,unwrap(angle.(fftpost)),color=:firebrick,label="High Mass",markersize=m)

    f[0, 1:2] = Label(f, "$moth", 
    fontsize = 24, 
    font = :bold)

    Legend(f[:,3],[l,h],["Low Mass","High Mass"])

    save("checks/JustFX/$moth.png",f)
end
##
# Plot Fx and Tz by flower position
for moth in collect(keys(allmoths))
    d = allmoths[moth]["data"]

    put_stim_in!(d,allmoths,moth)
    ##
    flowft = unique(select(d,:trial,:wb,:pos,:fx,:tz))
    clean = combine(groupby(flowft,[:trial,:wb])) do gdf 
        (
            pos = gdf.pos[1],
            fx = gdf.fx[1],
            tz = gdf.tz[1],
        )
    end
    ##
    pr = clean[clean.trial.=="pre",:]
    po = clean[clean.trial.=="post",:]
    f = Figure(size=(800,800))
    ax1 = Axis(f[1,1],title="FX Pre")
    ax2 = Axis(f[1,2],title="FX Post")
    ax3 = Axis(f[2,1],title="Yaw Pre")
    ax4 = Axis(f[2,2],title="Yaw Post")

    scatter!(ax1,pr.pos,pr.fx,color=:steelblue)
    scatter!(ax2,po.pos,po.fx,color=:firebrick)

    scatter!(ax3,pr.pos,pr.tz,color=:steelblue)
    scatter!(ax4,po.pos,po.tz,color=:firebrick)

    f[0, 1:2] = Label(f, "$moth", 
    fontsize = 24, 
    font = :bold)
    save("checks/ForceByPos/$moth.png",f)
end
##
# Lets look at mean ws Fx and Tz, by Position and Velocity 
for moth in collect(keys(allmoths))
    d = allmoths[moth]["data"]
    if  !in("vel",names(d))
        put_stim_in!(d,allmoths,moth)
    end
    ##
    flowft = unique(select(d,:trial,:wb,:pos,:fx,:tz,:vel))
    clean = combine(groupby(flowft,[:trial,:wb])) do gdf 
        (
            pos = gdf.pos[1],
            fx = gdf.fx[1],
            tz = gdf.tz[1],
            vel = gdf.vel[1],
        )
    end
    ##
    pr = clean[clean.trial.=="pre",:][2:end,:]
    po = clean[clean.trial.=="post",:][2:end,:]

    f = Figure(size=(1200,800))
    ax1 = Axis(f[1,1],title="Pre Flower Position")
    ax2 = Axis(f[1,2],title="Pre Flower Velocity")
    ax3 = Axis(f[2,1],title="Post Flower Position")
    ax4 = Axis(f[2,2],title="Post Flower Veloctiy")

    atz = Axis(f[1,1])
    hidedecorations!(atz)
    tz = lines!(atz,-1 .* pr.tz,color=:purple,alpha=0.7)
    atz = Axis(f[1,1])
    hidedecorations!(atz)
    fpos = lines!(atz,pr.pos,color=:forestgreen)
    fx = lines!(ax1,pr.fx,color=:firebrick,alpha=0.7)

    atz = Axis(f[1,2])
    hidedecorations!(atz)
    tz = lines!(atz,-1 .* pr.tz,color=:purple,alpha=0.7)
    atz = Axis(f[1,2])
    hidedecorations!(atz)
    fvel = lines!(atz,pr.vel,color=:gold3,alpha=0.7)
    fx = lines!(ax2,pr.fx,color=:firebrick,alpha=0.7)

    atz = Axis(f[2,1])
    hidedecorations!(atz)
    tz = lines!(atz,-1 .* po.tz,color=:purple,alpha=0.7)
    atz = Axis(f[2,1])
    hidedecorations!(atz)
    lines!(atz,po.pos,color=:forestgreen)
    fx = lines!(ax3,po.fx,color=:firebrick,alpha=0.7)

    atz = Axis(f[2,2])
    hidedecorations!(atz)
    tz = lines!(atz,-1 .* po.tz,color=:purple,alpha=0.7)
    atz = Axis(f[2,2])
    hidedecorations!(atz)
    lines!(atz,po.vel,color=:gold3,alpha=0.7)
    fx = lines!(ax4,po.fx,color=:firebrick,alpha=0.7)


    Legend(f[:,3],[fx,tz,fpos,fvel],["Fx","Yaw","Flower Pos","Flower Vel"])
    f[0, 1:2] = Label(f, "$moth", 
    fontsize = 24, 
    font = :bold)
    save("checks/fxtzposvel/$moth.png",f)
end

##
# Lets check coherence at each of these pos and velo 

for moth in collect(keys(allmoths))
    pr,po,vr,vo = transfer_function_coherence(allmoths,moth)
    f = Figure(size=(1200,600))
    ax1 = Axis(f[1,1],xscale=log10,title="Coherence of Position",ylabel="coherence",limits=(nothing,nothing,0,1))
    ax2 = Axis(f[1,2],xscale=log10,title="Coherence of Velocity",limits=(nothing,nothing,0,1))

    scatter!(ax1,freqqs,pr,color=:steelblue,markersize=15)
    scatter!(ax1,freqqs,po,color=:firebrick,markersize=15)

    scatter!(ax2,freqqs,vr,color=:steelblue,markersize=15)
    scatter!(ax2,freqqs,vo,color=:firebrick,markersize=15)

    f[0, 1:2] = Label(f, "$moth", 
    fontsize = 24, 
    font = :bold)

    save("checks/coh/$moth.png",f)
end
##
# Time Invariance of the FX Response

moth = "2024_11_08"
for moth in collect(keys(allmoths))
    pr = allmoths[moth]["fxpre"]
    po = allmoths[moth]["fxpost"]

    buff = zeros(Int(5e4))

    firpr = vcat(pr[1:Int(5e4)],buff)
    laspr = vcat(buff,pr[Int(5e4)+1:end])

    firpo = vcat(po[1:Int(5e4)],buff)
    laspo = vcat(buff,po[Int(5e4)+1:end])

    fr = fftfreq(Int(1e5),1e4)[2:201]

    fpr = abs.(fft(pr)[2:201])
    fpo = abs.(fft(po)[2:201])

    ffirpr = abs.(fft(firpr)[2:201])
    flaspr = abs.(fft(laspr)[2:201])

    ffirpo = abs.(fft(firpo)[2:201])
    flaspo = abs.(fft(laspo)[2:201])

    f = Figure(size=(800,1200)) 
    ax1 = Axis(f[1,1],title="All Pre",xscale=log10)
    ax2 = Axis(f[2,1],title="First Pre",xscale=log10)
    ax3 = Axis(f[3,1],title="Last Pre",xscale=log10)
    ax4 = Axis(f[1,2],title="All Post",xscale=log10)
    ax5 = Axis(f[2,2],title="First Post",xscale=log10)
    ax6 = Axis(f[3,2],title="Last Post",xscale=log10)

    lines!(ax1,fr,fpr,color=:steelblue)
    lines!(ax2,fr,ffirpr,color=:steelblue)
    lines!(ax3,fr,flaspr,color=:steelblue)

    lines!(ax4,fr,fpo,color=:firebrick)
    lines!(ax5,fr,ffirpo,color=:firebrick)
    lines!(ax6,fr,flaspo,color=:firebrick)
    f[0, 1:2] = Label(f, "$moth", 
    fontsize = 24, 
    font = :bold)
    save("checks/time_inv/$moth.png",f)
ends
##
for moth in collect(keys(allmoths))
    dir = "/home/doshna/Desktop/FatMothMuri/$moth"
    if !isdir(dir)
        mkdir(dir)
    end

    d = allmoths[moth]["data"]
    select!(d, Not(r"^[ft][xyz]_pc[4-9]|[ft][xyz]_pc10$"))
    save(dir*"/wing_beat_data.csv",d)

    itp1 = interpolate(allmoths[moth]["stimpre"], BSpline(Linear()))
    itp2 = interpolate(allmoths[moth]["stimpost"], BSpline(Linear()))
    itp3 = interpolate(allmoths[moth]["velpre"], BSpline(Linear()))
    itp4 = interpolate(allmoths[moth]["velpost"], BSpline(Linear()))


    ts = DataFrame(
        "fxpre" => allmoths[moth]["fxpre"],
        "fxpost" => allmoths[moth]["fxpost"],
        "stimpre" => itp1(LinRange(1, length(itp1), Int(1e5))),
        "stimpost"=> itp2(LinRange(1, length(itp1), Int(1e5))),
        "velpre" => itp3(LinRange(1, length(itp1), Int(1e5))),
        "velpost" => itp4(LinRange(1, length(itp1), Int(1e5)))
    )

    save(dir*"/time_series_data.csv",ts)
end
##
dir = "/home/doshna/Desktop/FatMothMuri/"
mc = get_mean_changes(allmoths)

save(dir*"/Moth_MetaData.csv",mc)