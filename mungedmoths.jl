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
# using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:steelblue,:firebrick],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
@load "fat_moths_set_1.jld" allmoths
fs = 1e4 
##
figs = false
##
moths = collect(keys(allmoths))
all_data = DataFrame()
cold_moths = ["2024_08_01","2024_06_06","2024_06_20","2024_06_24","2024_12_04_2","2024_12_03"]
moths = [m for m in moths if !in(m,cold_moths)]
##
for moth in moths 
    println(moth)
    d = allmoths[moth]
    if figs 
        f = get_tracking_fig(d["fxpre"],d["fxpost"])
        save("FatMothSet1/mungedfigs/tracking/$moth.png",f,px_per_unit=4)
        #Freq
        plot = data(d["data"])*mapping(:wbfreq,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
        fig = draw(plot,axis=(;))
        save("FatMothSet1/mungedfigs/freq/$moth.png",fig,px_per_unit=4)
        ## z forces
        plot = data(d["data"])*mapping(:fz,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
        fig = draw(plot,axis=(;))
        save("FatMothSet1/mungedfigs/zforce/$moth.png",fig,px_per_unit=4)
        ## Muscle Activities
        plot = data(d["data"]) * mapping(:phase,row=:muscle,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*AlgebraOfGraphics.density()*visual(alpha=0.5)
        fig = draw(plot,axis=(;))
        save("FatMothSet1/mungedfigs/muscle/$moth.png",fig,px_per_unit=4)
    end
    notpc = filter(col -> !contains(string(col), "_pc"), names(d["data"]))
    notpc = [x for x in notpc if x ∉ ["wbtime","pos","vel"]]
    # println(tmp)
    # put_stim_in!(d["data"],allmoths,moth)
    all_data=vcat(all_data,d["data"][!,notpc])
end
##

## Lets Plot the Freq and Muscle Munged Togetehr 

plot = data(all_data) * mapping(:phase,row=:muscle,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=250,normalization=:probability)*visual(alpha=0.5) 
f = draw(plot,figure=(; resolution=(600,1200)),axis=(;))
save("FatMothSet1/mungedfigs/AllMuscles.png",f,px_per_unit=4)
f
##
plot = data(all_data)*mapping(:wbfreq,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
d = draw(plot,axis=(;))
save("FatMothSet1/mungedfigs/ALLWBFreq.png",d,px_per_unit=4)
d
##
plot = data(all_data)*mapping(:fz,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]),layout=:moth)*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
g = draw(plot,axis=(;))
##
mean_changes = get_mean_changes(allmoths)
##

##
plot = data(mean_changes)*mapping(:mean_fz => "Relative Change in FZ",:mean_gain => "Average Change in Gain (%)",color=:moth)*visual(markersize=20,color=:cyan)
f = draw(plot,axis=(; xlabel = "Relative Change in Fz"))
# save("FatMothSet1/fzvsgainchange.png",f,px_per_unit=4)
f
##
"""
Looking In to Binning by flower position, Yaw torque, fx 
"""
df = add_relpos_column!(all_data)
df = add_relfx_column!(df)
df = add_relyaw_column!(df)

##
sa = df[df.muscle.=="rsa" .&& df.moth.=="2024_11_08",:]
select!(sa,:wb,:trial,:muscle,:time,:relfx,:relpos,:relyaw,:fx,:tz,:pos)
ssa = stack(sa,[:relyaw,:relfx,:relpos],variable_name=:axis,value_name=:rel)
ssa.axis = [s[4:end] for s in ssa.axis]

plot=data(ssa)*mapping(:time,color=:trial,row=:axis => renamer(["pos"=>"Flower Position","yaw"=>"Yaw T","fx"=>"Side Slip"]),col=:rel)*histogram(bins=100)|>draw
##
plot = data(sa)*mapping(:pos=>"Flower Position",:fx => "Side Slip Force",color=:trial)|>draw
plot = data(sa)*mapping(:pos=>"Flower Position",:tz => "Yaw Torque",color=:trial)|>draw

##
f = Figure()
ax1 = Axis(f[1,1])
ax2 = Axis(f[1,1])
hidedecorations!(ax2)
scatter!(ax1,sa[sa.trial.=="post",:wb],sa[sa.trial.=="post",:pos],label="pre",color=:gold)
scatter!(ax2,sa[sa.trial.=="post",:wb],sa[sa.trial.=="post",:fx],label="pre")
f
## wait okay hold on 
"""
Why is Binning by Flower Position Giving me sexy shit, but now with forces and torque? 
"""
pre,post = transfer_function(allmoths,"2024_11_08")

f = Figure()
ax = Axis(f[1,1],title="Gain Response",xlabel="Freq",ylabel="Gain (Force/position)",xscale=log10)
scatter!(freqqs,abs.(post),label="Post",markersize=15)
scatter!(freqqs,abs.(pre),label="Pre",markersize=15)
axislegend(ax,position=:lt)
ax2 =  Axis(f[1,2],title="Phase Response",xlabel="Freq",ylabel="Angle (rad)",xscale=log10)
scatter!(freqqs,angle.(post),label="Post",markersize=15)

scatter!(freqqs,angle.(pre),label="Pre",markersize=15)
axislegend(ax2,position=:lt)

f
##
pres = []
posts = []
sort!(mean_changes,:mean_gain)
for moth in mean_changes.moth
    pre,post = transfer_function(allmoths,moth)
    push!(pres,pre)
    push!(posts,post)
end
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
##
outdf = DataFrame()
for i in 1:8
    for j in 1:18
        tmp = Dict()
        tmp["moth"] = mean_changes.moth[i]
        phse = angle(posts[i][j]) - angle(pres[i][j])
        tmp["dif"] = phse
        tmp["freq"] = freqqs[j]
        push!(outdf,tmp,cols=:union)
    end
end
leftjoin!(outdf,mean_changes,on=:moth)
outdf = outdf[outdf.freq.<1.2,:]

##
outdf.freq = string.(outdf.freq)
plot = data(outdf)*mapping(:mean_fz,:dif,color=:freq)*visual(markersize=13) |> draw
##
"""
Lets Make the Pretty SICB FIgures 
"""
## Pretty Cairo Makie Themes 
using GLMakie
GLMakie.activate!()
sicbTheme = theme_minimal()
sicbTheme.textcolor=:black
sicbTheme.ytickfontcolor=:black
sicbTheme.fontsize= 20
sicbTheme.gridcolor=:lightcyan
sicbTheme.gridalpga=0.5
sicbTheme.palette = (color = [:steelblue,:firebrick],)
 
set_theme!(sicbTheme)
## gain by fz
x = mean_changes.mean_fz
y = mean_changes.mean_gain

regr = lm(@formula(mean_gain~mean_fz),mean_changes)
r = r2(regr)
fte = GLM.ftest(regr.model)

x_new = range(minimum(x),maximum(x),length=100)
pred = predict(regr, DataFrame(mean_fz = x_new),interval = :confidence, level = 0.95)

fig = Figure(resolution = (800,800))
ax = Axis(fig[1,1],
    xlabel = "Relative Change in Lift Force",
    ylabel = "Increase in Sensorimotor Gain (%)",
    xticklabelsize=20,
    yticklabelsize=20,
    limits = (-2,2,-30,100)
)

y_fake = range(minimum(y),maximum(y),length=100)
# lines!(ax,x_new,y_fake,color=:forestgreen,linewidth=8)
scatter!(ax,x,y,color=:royalblue3,markersize=20,alpha=0.7)
lines!(ax,x_new,convert(Vector{Float64},pred.prediction),
    linewidth=5,color=:lightskyblue3)
band!(ax, x_new, convert(Vector{Float64},pred.lower), convert(Vector{Float64},pred.upper),
    color = (:lightskyblue3, 0.2)
)
hlines!(ax,75,color=:grey,linewidth=8)

text!(ax, 
    "R² = $(round(r, digits=3))",
    position = (.7,-25),
    fontsize = 25
)
text!(ax, 
    "p = $(round(fte.pval, digits=3))",
    position = (.7,0),
    fontsize = 25
)
save("SICBFigs/GainFz.png",fig,px_per_unit=4)

display(fig)
## Changes in Wbfrequency 
# I want the moths in order of low fz to high fz
sort!(mean_changes,:mean_gain)
mean_changes.mothid = ["Moth $i" for i in 1:nrow(mean_changes)]



##

d4t = select(all_data,:moth,:wbfreq,:trial)
d4t.moth = [mean_changes[mean_changes.moth.==m,:mothid][1] for m in d4t.moth]
sort!(d4t,:moth)
##
sicbTheme.fontsize=30
set_theme!(sicbTheme)

plot = data(d4t)*mapping(:wbfreq => "Wing Beat Frequency (Hz)",
    color=:trial => renamer(["pre"=> "Low Mass", "post"=>"High Mass"])=>"",
    row=:moth)*AlgebraOfGraphics.density() 
f = draw(plot,figure=(; resolution=(800,1200)),
    axis=(; yticklabelsvisible=false,ylabel=""))

save("SICBFigs/wbfreq.png",f,px_per_unit=4)
f
##

d4t = select(all_data,:moth,:trial,:muscle,:phase)
idx = [m in ["ldlm","rdlm","ldvm","rdvm"] for m in d4t.muscle]
d4t = d4t[idx,:]
d4t.moth = [mean_changes[mean_changes.moth.==m,:mothid][1] for m in d4t.moth]
sort!(d4t,:moth)
##
plot = data(d4t)*mapping(:phase,row=:moth,col=:muscle,color=:trial)*
    AlgebraOfGraphics.density() |> draw
##
full_data.moth = [mean_changes[mean_changes.moth.==m,:mothid][1] for m in full_data.moth]
##
tmp = select(full_data,:moth,:trial,:lba1,:rba1,:lba2,:rba2,:ldlm1,:rdlm1)
tmp = transform(stack(tmp,[:lba1,:lba2],variable_name = "lsub",value_name="left"))
tmp = transform(stack(tmp,[:rba1,:rba2],variable_name = "rsub",value_name="right"))

tmp.ldif = tmp.left - tmp.ldlm1 
tmp.rdif = tmp.right - tmp.rdlm1 

tmp = transform(stack(tmp,[:ldif,:rdif]))
tmp.value = abs.(tmp.value) 
tmp[tmp.moth.=="Moth 8" .&& tmp.trial.=="pre",:value] .+= 0.01
##
select!(tmp,:moth,:trial,:value)
tmp = tmp[tmp.moth.!="Moth 5",:]
dropmissing!(tmp)
sort!(tmp,:moth)
plot = data(tmp)*mapping(:value => "BA - DLM Timing Offset (ms)",row=:moth,
    color=:trial => renamer(["pre"=> "Low Mass", "post"=>"High Mass"])=>"")*
    AlgebraOfGraphics.density() 
f = draw(plot,figure=(; resolution=(800,1200)),
    axis=(; yticklabelsvisible=false,ylabel=""))
save("SICBFigs/basalar.png",f,px_per_unit=4)
##

d = allmoths["2024_11_08"]["data"]
put_stim_in!(d,allmoths,"2024_11_08")
d = d[d.trial.=="pre",:]
##
d.time_abs .-= minimum(d.time_abs)

d = d[d.time_abs .< 0.5,:]

##
dposf = unique(select(d,:wb,:pos,:wbtime,:fx,:tz))
for n in names(dposf)
    dposf[!,n] = convert(Vector{Float64},dposf[!,n])
end
##
using CairoMakie
itp1 = interpolate(dposf.pos, BSpline(Linear()))
itp2 = interpolate(dposf.tz, BSpline(Linear()))
pos = itp1(LinRange(1, length(itp1), Int(25)))
tz = itp2(LinRange(1, length(itp2), Int(25)))

f = Figure(size=(600,1200),backgroundcolor=:transparent)

ax = Axis(f[1,1],ylabel="Flower",ylabelsize=20,limits=(0,0.5,500,560),backgroundcolor=:transparent)
lines!(ax,range(0,0.5,length=25),pos,color=:forestgreen,alpha=0.7,linewidth=5)
hidexdecorations!(ax)
hideydecorations!(ax,label=false)



a2 = Axis(f[2,1],ylabel="Side Slip",ylabelsize=20,backgroundcolor=:transparent)

lines!(a2,range(0,0.5,length=25),tz, color=:steelblue,alpha=0.7,linewidth=5)

hideydecorations!(a2,label=false)
hidexdecorations!(a2)
hidedecorations!(a4)
hidespines!(ax,:b)
hidespines!(a2,:b)
hidespines!(a3,:b)
hidespines!(a4,:b)
##
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
for (i,m) in enumerate(muscle_names)
    mus = d[d.muscle.==m,:time_abs]
    a = Axis(f[i+2,1],limits=(0,0.5,0,1),ylabel=m,backgroundcolor=:transparent)
    vlines!(mus,ymin = 0, ymax = 0.7, color=:black,alpha=0.7,linewidth=3)
    hidexdecorations!(a)
    hideydecorations!(a,label=false)
    hidespines!(a, :b)  # :b for bottom, can also use :l, :r, :t for left, right, top

end 
f
##

tzpre = allmoths["2024_11_08"]["ftpre"][:,6]
tzpost = allmoths["2024_11_08"]["ftpos"][:,6]

freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

ffttzpre = fft(tzpre)
ffttzpost = fft(tzpost)

fr = round.(fftfreq(length(tzpre),1e4),digits=4)

f = Figure()
ax = Axis(f[1,1],xscale=log10)
lines!(ax,fr[2:200],abs.(ffttzpre)[2:200],color=:steelblue)
lines!(ax,fr[2:200],abs.(ffttzpost)[2:200],color=:firebrick)
f
##
f = Figure(size=(600,1200))
a4 = Axis(f[1,1],limits=(0,0.5,0,1))
hidedecorations!(a4)

for (i, (start_idx, end_idx)) in enumerate(zip(dposf.wbtime[1:end-1], dposf.wbtime[2:end]))
    color_idx = i % 2 + 1  # Alternates between 1 and 2
    band!(a4, [start_idx, end_idx], [0, 0], [1, 1], color=colors[color_idx])
end

##

mc = get_mean_changes_yaw(allmoths)

##
tmp = select(all_data,:wb,:moth,:trial,:phase,:time,:muscle,:tz,:fx,:fz)

firsttime = unstack(select(tmp,Not(:phase)),[:wb,:moth,:trial,:tz,:fx,:fz],:muscle,:time,combine=first)
firstphase = unstack(select(tmp,Not(:time)),[:wb,:moth,:trial,:tz,:fx,:fz],:muscle,:phase,combine=first)
##

firsttime_long = dropmissing(stack(firsttime, Not(:wb,:trial,:moth,:tz,:fx),variable_name = :muscle,value_name=:time))

plot = data(firsttime_long)*mapping(:time,color=:trial, row =:muscle)* histogram(bins=100)|>draw
##
plot = data(all_data)*mapping(:wbfreq,color=:trial,layout=:moth)*histogram(bins=100)|>draw

firstphase = leftjoin(firstphase,mc,on=:moth)
firsttime = leftjoin(firsttime,mc,on=:moth)

##
increase = firsttime[firsttime.mean_gain .> 10 ,:]
first_phase_long = dropmissing(stack(increase, Not(:wb,:trial,:moth,:tz,:fx,:mass,:mean_gain,:mean_fz),variable_name = :muscle,value_name=:phase))
##
plot = data(first_phase_long)*mapping(:phase,color=:trial, row =:muscle)* histogram(bins=100)|>draw
##
increase.updownR = increase.rdlm - increase.rdvm
increase.updownL = increase.ldlm - increase.ldvm
increase.dlmoff = increase.ldlm - increase.rdlm

plot = data(dropmissing(select(increase,:trial,:moth,:updownR)))*mapping(:updownR,color=:trial)*histogram(bins=100)*visual(alpha=0.7) |> draw

longincrease = dropmissing(stack(select(increase,:moth,:trial,:updownR,:updownL),[:updownR,:updownL]))

plot = data(longincrease) * mapping(:value,color=:trial) *histogram(bins=100) |> draw

## 
inc = dropmissing(select(increase,:moth,:trial,:updownL,:updownR,:dlmoff))
dlmdvmvar = combine(groupby(inc,[:moth,:trial])) do gdf 
    (
        leftmean = mean(gdf.updownL),
        rightmean = mean(gdf.updownR),
        leftvar = var(gdf.updownL),
        rightvar = var(gdf.updownR),
        vardlm = var(gdf.dlmoff),
        meandlm = var(gdf.dlmoff)
    )
end

##
moths_ids = ["Moth $(i)" for i in 1:length(unique(mc.moth))]
mothref = DataFrame(:moth => sort(unique(mc.moth)), :moth_id => moths_ids)


lsaldlm = dropmissing(select(increase,:lsa,:ldlm,:moth,:wb,:fz,:trial,:ldvm))
rsardlm = dropmissing(select(increase,:rsa,:rdlm,:moth,:wb,:fz,:trial,:ldvm))
lsaldlm.sa = lsaldlm.lsa
lsaldlm.dlm = lsaldlm.ldlm
rsardlm.sa = rsardlm.rsa
rsardlm.dlm = rsardlm.rdlm
lsaldlm.diff = lsaldlm.lsa - lsaldlm.ldlm
rsardlm.diff = rsardlm.rsa - rsardlm.rdlm


lsaldlm.fz_flipped = -1 .* lsaldlm.fz
lsaldlm = leftjoin(lsaldlm,mothref,on=:moth)
bothsides = vcat(select(lsaldlm,:moth,:trial,:fz,:diff,:sa,:dlm),select(rsardlm,:moth,:trial,:fz,:diff,:sa,:dlm))
lsaldlm.diffms = lsaldlm.diff .* 1000
##
plot = data(lsaldlm)*mapping(:diffms,color=:trial=>renamer(["pre"=> "Before Feeding","post"=>"After Feeding"])=> "Condition")*AlgebraOfGraphics.histogram(bins=500,normalization=:probability)*visual(alpha=0.7) 
f = draw(plot,figure=(; size=(800,600)),axis=(; xlabel = "SA - DLM Offset (ms)",xlabelsize=25,ylabelsize=25,limits=(-5,10,nothing,nothing),ylabel = "Proportion"))
save("GoodFigs/SaDLMOff.png",f,px_per_unit=4)
f
##
plot = data(lsaldlm)*mapping(:fz_flipped => "Lift Force (N)",color=:trial=>renamer(["pre"=> "Before Feeding","post"=>"After Feeding"])=> "Condition",layout=:moth_id)*histogram(bins=100,normalization=:probability)
f = draw(plot,figure=(; size=(1000,800)),axis=(; xlabelsize=25,ylabelsize=25,aspect=1))
# save("GoodFigs/Z_Hist.png",f,px_per_unit=4)


##
ploto = data(lsaldlm)*mapping(:sa => "Subalar Spike Time (s)",:ldvm=>"DVM Spike Time (s)",color=:fz_flipped =>"Lift Force (N)",row=:trial=>renamer(["pre"=> "Before Feeding","post"=>"After Feeding"]),col=:moth_id) 
f = draw(ploto,figure=(; size=(1000,800)),axis=(; xlabelsize=25,ylabelsize=25,aspect=1),colorbar=(position=:bottom, size=25,),scales(Color = (; colormap = :Oranges)))
# save("GoodFigs/DVM_SA_ZForce.png",f,px_per_unit=4)
##

for moth in unique(lsaldlm.moth)
    println("for Moth $moth")
    for t in ["pre","post"]
        g = lsaldlm[lsaldlm.moth.==moth .&& lsaldlm.trial.==t,:]
        hh = lm(@formula(dlm~sa),g)
        println("$t is $(coef(hh)[2])")
    end
end
##
laxdlm = dropmissing(select(increase,:moth,:trial,:tz,:lax,:ldlm,:rax,:rdlm,:lba,:rba))
laxdlm.dlmdiff = laxdlm.ldlm - laxdlm.rdlm
laxdlm = leftjoin(laxdlm,mothref,on=:moth)
laxdlm = laxdlm[laxdlm.moth .!= "2025_03_20",:]
##

plot = data(laxdlm)*mapping(:dlmdiff => "Left - Right DLM Timing (s)",color=:tz => "Yaw Torque (Nmm)",row=:trial=>renamer(["pre"=> "Before Feeding","post"=>"After Feeding"]),col=:moth_id)
f = draw(plot,figure=(; size=(1200,800)),axis=(; xlabelsize=25,ylabelsize=25,xlabel="Wingbeat"),colorbar=(position=:bottom, size=25,),scales(Color = (; colormap = :thermal)))
save("GoodFigs/DLM_Tracking.png",f,px_per_unit=4)
f
##
    laxdlm = laxdlm[laxdlm.moth_id.!= "Moth 2",:]
    # laxdlm = laxdlm[laxdlm.lax .> 0.02,:]
    laxdlm.off = laxdlm.lba - laxdlm.lax
    laxdlm.laxdif = laxdlm.lax - laxdlm.rax
    plot = data(laxdlm)*mapping(:rax => "Left Axillary",color=:tz => "Yaw Torque (Nmm)",row=:trial=>renamer(["pre"=> "Before Feeding","post"=>"After Feeding"]),col=:moth_id) |> draw

##
moths_ids = ["Moth $(i)" for i in 1:length(unique(mc.moth))]
mothref = DataFrame(:moth => sort(unique(mc.moth)), :moth_id => moths_ids)


tmp = dropmissing(select(firstphase,:moth,:trial,:lsa,:ldlm,:ldvm,:fz))
tmp = leftjoin(tmp,mothref,on=:moth)

plot = data(tmp)*mapping(:lsa,:ldvm,color=:fz,layout=:moth_id) |> draw