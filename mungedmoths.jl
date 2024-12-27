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
# theme.palette = (color = [:turquoise,:coral],)
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
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
peaks = DataFrame()
for moth in moths 
    pre = allmoths[moth]["fxpre"]
    post = allmoths[moth]["fxpost"]

    fftpre = abs.(fft(pre)[2:50000])
    fftpost = abs.(fft(post)[2:50000])
    freqrange = round.(fftfreq(length(pre),fs)[2:50000],digits=4)
    d4t = all_data[all_data.moth.==moth,:]
    for f in freqqs
        id = findfirst(x -> x == f, freqrange)
        peakpre = fftpre[id]
        peakpost = fftpost[id]
        prdic = Dict("moth"=>moth,"freq" => f, "trial" => "pre", 
            "peak" => peakpre, "fz" => mean(d4t[d4t.trial.=="pre",:fz]) )
        podic = Dict("moth"=>moth,"freq" => f, "trial" => "post", 
            "peak" => peakpost,"fz" => mean(d4t[d4t.trial.=="post",:fz]) )
        push!(peaks,prdic,cols=:union)
        push!(peaks,podic,cols=:union)
    end
end
peaks.fz = peaks.fz .* -1 
##
##
g=9.81
ms = Dict(
    "2024_11_01" => Dict("pre"=>1.912,"post"=>2.075),
    "2024_11_04" => Dict("pre"=>2.149,"post"=>2.289),
    "2024_11_05" => Dict("pre"=>2.190,"post"=>2.592),
    "2024_11_07" => Dict("pre"=>1.801,"post"=>1.882),
    "2024_11_08" => Dict("pre"=>2.047,"post"=>2.369),
    "2024_11_11" => Dict("pre"=>1.810,"post"=>2.090),
    "2024_11_20" => Dict("pre"=>1.512,"post"=>1.784),
    "2024_12_03" => Dict("pre"=>2.728,"post"=>2.827),
    "2024_12_04_2" => Dict("pre"=>2.790,"post"=>3.456),
    "2024_08_01" => Dict("pre"=>2.796,"post"=>2.957),
    "2024_06_06" => Dict("pre"=>1.250,"post"=>1.658),
    "2024_06_20" => Dict("pre"=>2.090,"post"=>2.430),
    "2024_06_24" => Dict("pre"=>3.020,"post"=>3.411)

)
##
function normalize_fz(row, ms, g)
    return row.fz / ((ms[row.moth]["pre"]/1000) * g)
end

peaks.norm_fz = map(row -> normalize_fz(row, ms, g), eachrow(peaks))

##
grouped = groupby(peaks,[:moth,:freq])

changes = combine(grouped) do gdf 
    pre_vals = filter(r -> r.trial == "pre",gdf)
    post_vals = filter(r -> r.trial == "post",gdf)
    (
        fz_change =  (mean(post_vals.norm_fz) - mean(pre_vals.norm_fz)) / abs(mean(pre_vals.norm_fz)),
        gain_change = (mean(post_vals.peak) - mean(pre_vals.peak))/abs(mean(pre_vals.peak))*100
    )
end
##
f = gain_by_freq_fig(changes)
##

mean_changes = combine(groupby(changes, :moth),
   :fz_change => mean => :mean_fz,
   :gain_change => mean => :mean_gain
)
mean_changes.mass .= 0.
for row in eachrow(mean_changes)
    row.mass = ms[row.moth]["post"] - ms[row.moth]["pre"] 
end
##

##
plot = data(mean_changes)*mapping(:mean_fz => "Relative Change in FZ",:mean_gain => "Average Change in Gain (%)",color=:moth)*visual(markersize=20,color=:cyan)
f = draw(plot,axis=(; xlabel = "Relative Change in Fz"))
# save("FatMothSet1/fzvsgainchange.png",f,px_per_unit=4)
f
##
df = select(all_data,:moth,:wb,:trial,:muscle,:phase,:fz,:wbfreq,:time)
full_data = get_big_data(df,"time")
##
meanmus = combine(groupby(all_data,[:moth,:trial,:muscle]),
    :time => mean => :mean_time
    )

dmus = combine(groupby(meanmus,[:moth,:muscle])) do gdf 
    pre_vals = filter(r -> r.trial == "pre",gdf)
    post_vals = filter(r -> r.trial == "post",gdf)
    (
        time_change =  (mean(post_vals.mean_time) - mean(pre_vals.mean_time)),
        
    )
end
##


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
sicbTheme.fontsize= 30
sicbTheme.gridcolor=:lightcyan
sicbTheme.gridalpga=0.5
# sicbTheme.palette = (color = [:turquoise,:coral],)
 
set_theme!(sicbTheme)
## gain by fz
x = mean_changes.mean_fz
y = mean_changes.mean_gain

regr = lm(@formula(mean_gain~mean_fz),mean_changes)
r = r2(regr)

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
