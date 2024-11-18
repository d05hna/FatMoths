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
        plot = data(d["data"]) * mapping(:phase,row=:muscle,color=:trial => renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*histogram(bins=250,normalization=:probability)*visual(alpha=0.5)
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
##
grouped = groupby(peaks,[:moth,:freq])

changes = combine(grouped) do gdf 
    pre_vals = filter(r -> r.trial == "pre",gdf)
    post_vals = filter(r -> r.trial == "post",gdf)
    (
        fz_change =  (mean(post_vals.fz) - mean(pre_vals.fz)) / abs(mean(pre_vals.fz)) * 100,
        gain_change = (mean(post_vals.peak) - mean(pre_vals.peak))/abs(mean(pre_vals.peak)) * 100
    )
end
##
f = gain_by_freq_fig(changes)
##

mean_changes = combine(groupby(changes, :moth),
   :fz_change => mean => :mean_fz,
   :gain_change => mean => :mean_gain
)
##

mean_changes.mean_fz = mean_changes.mean_fz .* -1
##
true_fed_pct = Dict(
    "2024_11_01" => 9.3,
    "2024_11_04" => 6.94,
    "2024_11_05" => 18.38,
    "2024_11_07" => 4.48,
    "2024_11_08" => 17.98,
    "2024_11_11" => 15.47
)
##
fedf = DataFrame(moth = collect(keys(true_fed_pct)), fed = collect(values(true_fed_pct)))
leftjoin!(mean_changes,fedf,on=:moth)
mean_changes.fed = convert(Vector{Float64},mean_changes.fed)
##
f = Figure()
ax = Axis(f[1,1],xlabel= "Change in Average Wing Beat Z Force (%)",ylabel="Average Increase in Gain (%)")
gchange=scatter!(ax,mean_changes.mean_fz,mean_changes.mean_gain,markersize=20,color=:coral)
mchange=scatter!(ax,mean_changes.mean_fz,mean_changes.fed,color=:green,markersize=15)
Legend(f[1,2],[gchange,mchange],["True Gain Change","Explained by Mass Alone"])
save("FatMothSet1/fzvsgainchangevsmass.png",f,px_per_unit=4)
f
##

df = select(all_data,:moth,:wb,:trial,:muscle,:phase)
full_data = get_big_data(df)
##
save("highdimensionbigdata.csv",full_data)
##
using UMAP 
##
# umap_model = UMAP.UMAP()
x = Matrix(full_data[!,Not(:wb,:moth,:trial)])

mod = UMAP_(x)