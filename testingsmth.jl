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
using Dierckx
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
sicbTheme = theme_minimal()
sicbTheme.textcolor=:black
sicbTheme.ytickfontcolor=:black
sicbTheme.fontsize= 30
sicbTheme.gridcolor=:lightcyan
sicbTheme.gridalpga=0.5
set_theme!(sicbTheme)

##
@load "fat_moths_set_1.jld" allmoths
fs = 1e4 

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
        ws_start_time = minimum(group_df.time_abs)
        if nrow(ldlm_rows) > 0
            mot = ldlm_rows.moth[1]
            tri = ldlm_rows.trial[1]
        else
            mot = missing
            tri = missing
        end

        # Return named tuple with results
        (
            time_difference = time_diff,
            average_time_abs = avg_time_abs,
            wstime = ws_start_time,
            fx = mean(group_df.fx),
            tz = mean(group_df.tz),
            m1count = nrow(ldlm_rows),
            m2count = nrow(rdlm_rows),
            moth = mot,
            trial = tri

        )
    end
    # leftjoin!(results,unique(select(df,:wb,:trial)),on=:wb)
    return results
end

function get_df_dlms(moth,allmoths)
    df = allmoths[moth]["data"]
    wdf = select(df,:wb,:time_abs,:muscle,:time,:trial)
    dlms = wdf[wdf.muscle .=="ldlm" .|| wdf.muscle .== "rdlm",:]

    diffs = analyze_muscle_timing(dlms,"ldlm","rdlm")
    dropmissing!(diffs)
    pred = diffs[diffs.trial.=="pre",:]

    posd = diffs[diffs.trial.=="post",:]

    pred.wb .-= minimum(pred.wb)
    posd.wb .-= minimum(posd.wb)

    pred.score = zscore(pred.time_difference)
    posd.score = zscore(posd.time_difference)

    fu = vcat(pred,posd,cols=:union)
    fu = fu[fu.score .> -2 .&& fu.score .<2,:]
    # println(first(fu,4))
    # select!(fu,:trial,:wb,:time_difference,:score)
    fu.moth .= moth
    return(fu)
end

function get_ccs(points,moth,allmoths)
    working = points[points.moth.==moth,:]
    stimpre = float.(allmoths[moth]["stimpre"])
    stimpost = float.(allmoths[moth]["stimpost"])

    pretimes = working[working.trial.=="pre",:average_time_abs]
    pretimes .-= minimum(pretimes)

    postimes = working[working.trial.=="post",:average_time_abs]
    postimes .-= minimum(postimes)

    pred = working[working.trial.=="pre",:time_difference]
    posd = working[working.trial.=="post",:time_difference]

    spre = Spline1D(pretimes,pred,k=3)
    spost = Spline1D(postimes,posd,k=3)

    t = range(0,10,length=Int(3000))
    pred_long = spre.(t)
    posd_long = spost.(t)

    maxlag = Int(600)
    lagtime = (-maxlag:maxlag) / 300

    ccpre = crosscor(pred_long,stimpre,-maxlag:maxlag;demean=true)
    ccpost = crosscor(posd_long,stimpost,-maxlag:maxlag;demean=true)

    f = Figure()
    ax = Axis(f[1,1],
    xlabel="lag time (s)",
    ylabel = "Correlation",
    title="DLM Offset and Stimulus Cross Correlation"
    )
    re = lines!(ax,lagtime,ccpre)
    po = lines!(ax,lagtime,ccpost)

    Legend(f[1,2],[re,po],["pre","post"])
    # save("FatMothSet1/CrossCor/FORCEDLMS_$(moth).png",f,px_per_unit=4)

    shiftmin = (argmin(ccpost) - argmin(ccpre))/300

    return(shiftmin)
end

function get_laxs(moth,allmoths)
    stimpre = float.(allmoths[moth]["stimpre"])
    stimpost = float.(allmoths[moth]["stimpost"])
    d = allmoths[moth]["data"]
    
    res = get_big_data(d,"time")
    
    prd = res[res.trial.=="pre",[:moth,:trial,:lax1]]
    dropmissing!(prd)
    pos = res[res.trial.=="post",[:moth,:trial,:lax1]]
    dropmissing!(pos)
    
    
    pred = prd.lax1
    posd = pos.lax1
    
    
    # pred = pred[pred.>0.01]
    
    
    spre = Spline1D(range(0,10,length=length(pred)),pred,k=3)
    spost = Spline1D(range(0,10,length=length(posd)),posd,k=3)
    t = range(0,10,length=Int(3000))
    predlong = spre.(t)
    posdlong = spost.(t)
    
    maxlag = Int(600)
    ccpre = crosscor(predlong,stimpre,-maxlag:maxlag;demean=true)
    ccpost = crosscor(posdlong,stimpost,-maxlag:maxlag;demean=true)

    f = Figure()
    ax1 = Axis(f[1,1],title="Time Pre")
    ax2 = Axis(f[2,1],title="Time Post")
    lines!(ax1,zscore(predlong),color=:blue)
    lines!(ax1,zscore(stimpre),color=:red,alpha=0.7)
    lines!(ax2,zscore(posdlong),color=:blue)
    lines!(ax2,zscore(stimpost),color=:red,alpha=0.7)
    lt = (-maxlag:maxlag)/300

    ax3 = Axis(f[1,2],title="CC Pre")
    lines!(ax3,lt,ccpre)
    ax4=Axis(f[2,2],title="CC Post")
    lines!(ax4,lt,ccpost)
    save("FatMothSet1/CrossCor/ForceLAX$moth.png",f,px_per_unit=4)

    shift = (argmax(ccpost) - argmax(ccpre))/300
    return shift

end


##
moths = collect(keys(allmoths))
bad_moths = ["2024_08_01","2024_06_06","2024_06_20","2024_06_24","2024_12_04_2","2024_12_03"]
moths = [m for m in moths if !in(m,bad_moths)]
##
all_dlms = DataFrame()
for moth in moths
    d = get_df_dlms(moth,allmoths)
    all_dlms = vcat(all_dlms,d,cols=:union)
end
##
dlmshifts = DataFrame()
for moth in moths 
    s = get_ccs(all_dlms,moth,allmoths)
    tmp = Dict("moth"=>moth,"DLMshift"=>s)
    push!(dlmshifts,tmp,cols=:union)
end
##
laxshifts = DataFrame()
for moth in moths 
    s = get_laxs(moth,allmoths)
    tmp = Dict("moth"=>moth,"LAXshift"=>s)
    push!(laxshifts,tmp,cols=:union)
end
leftjoin!(laxshifts,mean_changes,on=:moth)
##
shifts = leftjoin(dlmshifts,laxshifts,on=:moth)
rename!(shifts,["DLMshift"=> "DLM", "LAXshift"=>"AX"])
tmp = transform(stack(shifts,[:DLM,:AX],variable_name = "muscle",value_name="shift"))
##

regrlax = lm(@formula(LAXshift~mean_gain),laxshifts)
rlax = r2(regrlax)

regrdlm = lm(@formula(DLMshift~mean_gain),dlmshifts)
rdlm = r2(regrdlm)

x = dlmshifts.mean_gain
x_new = x_new = range(minimum(x),maximum(x),length=100)

predlax = predict(regrlax, DataFrame(mean_gain = x_new),interval = :confidence, level = 0.95)
preddlm = predict(regrdlm,DataFrame(mean_gain = x_new),interval = :confidence,level=0.95)


fig = Figure(resolution = (1200,800))
ax = Axis(fig[1,1],
    xlabel = "% Change in Sensorimotor Gain",
    ylabel = "Change in Phase Relative To Stimulus (s)",
    xticklabelsize=20,
    yticklabelsize=20,
)

axl = scatter!(ax,laxshifts.mean_gain,laxshifts.LAXshift,color=:firebrick,markersize=20,alpha=0.7)
dl = scatter!(ax,dlmshifts.mean_gain,dlmshifts.DLMshift,color=:royalblue,markersize=20,alpha=0.7)
lines!(ax,x_new,convert(Vector{Float64},predlax.prediction),
    linewidth=5,color=:crimson)
band!(ax, x_new, convert(Vector{Float64},predlax.lower), convert(Vector{Float64},predlax.upper),
    color = (:crimson, 0.2)
)
lines!(ax,x_new,convert(Vector{Float64},preddlm.prediction),
    linewidth=5,color=:skyblue2)
band!(ax, x_new, convert(Vector{Float64},preddlm.lower), convert(Vector{Float64},preddlm.upper),
    color = (:skyblue2, 0.2)
)
# text!(ax, 
#     "R² = $(round(rlax, digits=3))",
#     position = (25,-2.7),
#     fontsize = 25,
#     color=:red
# )
# text!(ax, 
#     "R² = $(round(rdlm, digits=3))",
#     position = (25,-3),
#     fontsize = 25,
#     color=:blue
# )
Legend(fig[1,2],[dl,axl],["DLM","3AX"],title="Muscle")
save("FatMothSet1/AXDLMPHASEGAIN.png",fig,px_per_unit=4)
fig

##

t = 1/300:1/300:10
fig = Figure()
ax1 = Axis(fig[1,1],title="Pre")
ax2 = Axis(fig[2,1],title="Post")

lines!(ax1,t,zscore(stimpre),color=:red,alpha=0.7)
lines!(ax1,t,zscore(pred_long),color=:blue)

s = lines!(ax2,t,zscore(stimpost),color=:red,alpha=0.7)
d = lines!(ax2,t,zscore(pos_long),color=:blue)

Legend(fig[1,2],[s,d],["Stimulus","DLM Timing Offset"])
save("FatMothSet1/$(m)_DlMTiming_stim.png",fig,px_per_unit=4)
fig


##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300,1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30]


# fx = allmoths[moth]["fxpre"]
freqrange = fftfreq(Int(10*fs),fs)[2:120]

fftstimpre = (fft(stimpre))[2:120]
fftstimpost = (fft(stimpost))[2:120]
fftpre = (fft(predlong))[2:120]
fftpost = (fft(posdlong))[2:120]

cros_pre = fftpre .* conj(fftstimpre)
cros_pos = fftpost .* conj(fftstimpost)
##
fig = Figure(resolution=(1200,600))
ax = Axis(fig[1,1],xscale=log10,ylabel="Amplitude",xlabel="Frequency",title="lax1",limits=(nothing,12,nothing,nothing))
# ax2 = Axis(fig[2,1],xscale=log10,ylabel="Amplitude",xlabel="Frequency",title="stim",limits=(nothing,12,nothing,nothing))
# ax3 = Axis(fig[3,1],xscale=log10,ylabel="Amplitude",xlabel="Frequency",limits=(nothing,12,nothing,nothing))
lines!(ax,freqrange,abs.(fftpre),color=:red)
lines!(ax,freqrange,abs.(fftpost),color=:blue)

# lines!(ax2,freqrange,abs.(fftstimpre),color=:blue)
# hi = lines!(ax3,freqrange,abs.(fftpre./fftstimpre),color=:green)
# lo = lines!(ax,freqrange,abs.(fftstimpost./ fftpost),color=:red)
dr = vlines!(ax,freqqs,color=:grey,linewidth=3,alpha=0.3)
dr = vlines!(ax2,freqqs,color=:grey,linewidth=3,alpha=0.3)
dr = vlines!(ax3,freqqs,color=:grey,linewidth=3,alpha=0.3)

# Legend(fig[1,2],[lo,hi,dr],["Low Mass","High Mass","Flower Frequency"])
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
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300]# 5.300, 6.100, 7.900, 8.900, 11.30]
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
##
moth = "2024_11_20"
l = allmoths[moth]["data"]
thisguy = analyze_muscle_timing(l,"rax","lax")
dropmissing!(thisguy)
# thisguy = all_dlms[all_dlms.moth.==moth,:]
preguy = thisguy[thisguy.trial.=="pre",:]
postguy = thisguy[thisguy.trial.=="post",:]

preguy.wstime .-= minimum(preguy.wstime)
postguy.wstime .-= minimum(postguy.wstime)

preguy.wstime = round.(preguy.wstime,digits=4)
postguy.wstime = round.(postguy.wstime,digits=4)

fs = 1e4

preguylong = zeros(Int(10*fs))

for i in 1:nrow(preguy)-1
    start_idx = Int(floor(fs*preguy.wstime[i]))+1
    en_idx = Int(ceil(fs*preguy.wstime[i+1]))
    preguylong[start_idx:en_idx] .= preguy.time_difference[i]
end
last_idx = Int(ceil(fs * preguy.wstime[end]))
preguylong[last_idx:end] .= preguy.time_difference[end]

postguylong = zeros(Int(10*fs))

for i in 1:nrow(postguy)-1
    start_idx = Int(floor(fs*postguy.wstime[i]))+1
    en_idx = Int(ceil(fs*postguy.wstime[i+1]))
    postguylong[start_idx:en_idx] .= postguy.time_difference[i]
end
last_idx = Int(ceil(fs * postguy.wstime[end]))
postguylong[last_idx:end] .= postguy.time_difference[end]

fftpre = fft(preguylong)[2:50]
fftpost= fft(postguylong)[2:50]
fr = fftfreq(Int(10*fs),fs)[2:50]


fig = Figure(resolution=(1200,600))


ax = Axis(fig[1,1],xscale=log10,xlabel="Frequency",ylabel="Amplitude")
r = lines!(ax,fr,abs.(fftpre),color=:steelblue,linewidth=4)
a = lines!(ax,fr,abs.(fftpost),color=:firebrick,linewidth=4)
l = vlines!(ax,freqqs,color=:grey,linewidth=3,alpha=0.3)
Legend(fig[1,2],[r,a,l],["Light Response","Heavy Response","Flower Frequency"])
save("FatMothSet1/StepFFT/$(moth)AX.png",fig,px_per_unit=4)
fig
##
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
using ColorTypes
dlmcolor = RGB(114/225,74/225,71/225)
axcolor = RGB(138/255,159/255,116/255)
fatred = RGB(237/255,51/255,56/255)
skinnyblue = RGB(51/255,122/255,189/255)
##
thisguy = all_dlms[all_dlms.moth.=="2024_11_08",:]
preguy = thisguy[thisguy.trial.=="pre",:]
preguy.time = preguy.average_time_abs .- minimum(preguy.average_time_abs)
stimpre = allmoths["2024_11_08"]["stimpre"]
##
fig = Figure(resolutio=(800,800))
ax = Axis(fig[1,1],yticklabelsvisible=false,xlabel = "Time (s)",limits=(0,10,-3,3))
lines!(ax,range(0,10,length=length(stimpre)),zscore(float.(stimpre)),color=:skyblue,alpha=0.5,linewidth=6)
# scatter!(ax,preguy.time,zscore(preguy.time_difference),color=dlmcolor,markersize=12)

save("SICBFigs/PreStimPLANE.png",fig,px_per_unit=4)
fig
##
dl = preguy.time_difference
t = preguy.time

posguy = thisguy[thisguy.trial.=="post",:]
posguy.time = posguy.average_time_abs .- minimum(posguy.average_time_abs)
stimpost = allmoths["2024_11_08"]["stimpost"]

##

spre = Spline1D(t,dl,k=3)
spost = Spline1D(posguy.time,posguy.time_difference)

tn = range(0,10,length=Int(3000))

pred_long = spre.(tn)
posd_long = spost.(tn)

maxlag = Int(900)
lagtime = (-maxlag:maxlag) / 300

ccpre = crosscor(pred_long,float.(stimpre),-maxlag:maxlag;demean=true)
ccpost = crosscor(posd_long,float.(stimpost),-maxlag:maxlag;demean=true)

##
fig = Figure(resolution = (800,800))
ax = Axis(fig[1,1],xlabel="Time Lag (s)",ylabel = "Correlation")

lines!(ax,-1 .*lagtime,ccpre,color = skinnyblue,linewidth=5 )
lines!(ax,-1 .*lagtime,ccpost,color=fatred,linewidth=5)
save("SICBFigs/1108DLMCrosscor.png",fig,px_per_unit=4)
fig
##
regrlax = lm(@formula(AX~mean_gain),shifts)
rlax = r2(regrlax)

regrdlm = lm(@formula(DLM~mean_gain),shifts)
rdlm = r2(regrdlm)
##
x = shifts.mean_gain
x_new = x_new = range(minimum(x),maximum(x),length=100)

predlax = predict(regrlax, DataFrame(mean_gain = x_new),interval = :confidence, level = 0.95)
preddlm = predict(regrdlm,DataFrame(mean_gain = x_new),interval = :confidence,level=0.95)
##
fig = Figure(resolution=(800,800))
ax = Axis(fig[1,1],xlabel="Percent Change in Gain",ylabel="Shift Relative to Stimulus",
    yticklabelsvisible=false,limits=(nothing,nothing,-2,3))

scatter!(ax,shifts.mean_gain,shifts.AX,color=axcolor,markersize=20,alpha=0.7)
scatter!(ax,shifts.mean_gain,shifts.DLM,color=dlmcolor,markersize=20,alpha=0.7)

lines!(ax,x_new,convert(Vector{Float64},preddlm.prediction),
    linewidth=5,color=dlmcolor)
band!(ax, x_new, convert(Vector{Float64},preddlm.lower), convert(Vector{Float64},preddlm.upper),
    color = (dlmcolor, 0.2)
)
lines!(ax,x_new,convert(Vector{Float64},predlax.prediction),
    linewidth=5,color=axcolor)
band!(ax, x_new, convert(Vector{Float64},predlax.lower), convert(Vector{Float64},predlax.upper),
    color = (axcolor, 0.2)
)

save("SICBFigs/CrossCorVGainBoth.png",fig,px_per_unit=4)
fig