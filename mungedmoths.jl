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

cold_moths = ["2024_08_01","2024_06_06","2024_06_20","2024_06_24"]
mean_changes.group = ifelse.(in.(mean_changes.moth, Ref(cold_moths)), "cold", "ether")
##
fedf = DataFrame(moth = collect(keys(true_fed_pct)), fed = collect(values(true_fed_pct)))
leftjoin!(mean_changes,fedf,on=:moth)
mean_changes.fed = convert(Vector{Float64},mean_changes.fed)
##
plot = data(mean_changes)*mapping(:mean_fz => "Relative Change in FZ",:mean_gain => "Average Change in Gain (%)",color=:group)*visual(markersize=20,color=:cyan)
f = draw(plot,axis=(; xlabel = "Relative Change in Fz"))
# save("FatMothSet1/fzvsgainchange.png",f,px_per_unit=4)
f
##

df = select(all_data,:moth,:wb,:trial,:muscle,:phase,:fz,:wbfreq,:time)
full_data = get_big_data(df,"time")
##
save("/home/doshna/Documents/PHD/FatMothMLProject/highdimensionbigdata.csv",full_data)
##
counts = [x for x in names(full_data) if occursin("count",x)]
push!(counts,"trial","moth")

countdf = select(full_data,counts)
select!(countdf,:moth,:trial,:)

longcount = stack(countdf,Not(:moth,:trial),variable_name = "muscle",value_name = "count")

longcount.count = Int.(longcount.count)

longcount.muscle= [x[1:end-6] for x in longcount.muscle]
##

plot = data(longcount) * mapping(:count,row=:muscle,color=:trial => 
    renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*
    histogram(bins=7,normalization=:probability) * visual(alpha=0.5) |> draw
##
dvm = longcount[longcount.muscle .== "ldvm",:]

plot = data(dvm) * mapping(:count,row=:moth,color=:trial => 
    renamer(["pre"=>"Before Feeding","post"=>"After Feeding"]))*
    histogram(bins=7,normalization=:probability) * visual(alpha=0.5) |> draw

##
"""
Does the Distribution of Spike timings (Relative to Other Spikes) Change in Mean or in Variance Depending on Z Force (A Heuristic for Gain Increase)
"""
## Just so I have it pull out the absolute changes in fz 
fzs = unique(select(peaks,:moth,:fz,:trial))
grouped = groupby(fzs,[:moth])

fzchanges = combine(grouped) do gdf 
    pre_vals = filter(r -> r.trial == "pre",gdf)
    post_vals = filter(r -> r.trial == "post",gdf)
    (
        fz_change =  (mean(post_vals.fz) - mean(pre_vals.fz)),
    )
end
##

firstphase = filter(str -> !in(str[end], ['2','3','4','5','t']), names(full_data))
firstspike = select(full_data,firstphase)

rename!(firstspike,["moth","wb","trial","fz","wbfreq","lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"])
## Lets Look at DLMS only first bc this is a lot of pwise combinations

powers = select(firstspike,:moth,:trial,:fz,:wbfreq,:ldlm,:rdlm,:ldvm,:rdvm)
dropmissing!(powers)

powers.dlmdiff = powers.ldlm - powers.rdlm
powers.dvmdiff = powers.ldvm - powers.rdvm
powers.ldlmdvm = powers.ldlm - powers.ldvm
powers.rdlmdvm = powers.rdlm - powers.rdvm


# save("powermusclestimings.csv",powers)
##
dlm_difs = select(powers,:moth,:trial,:dlmdiff)

gdf = groupby(dlm_difs, [:moth, :trial])
dlm_vars = combine(gdf, :dlmdiff => var => :variance)
var_ratio = combine(groupby(dlm_vars, :moth)) do group
    (variance_ratio = (group[group.trial .== "post", :variance][1] - 
                     group[group.trial .== "pre", :variance][1]),)
 end

leftjoin!(mean_changes,var_ratio,on=:moth)
##
plot = data(mean_changes)*mapping(:mean_gain => "Relative Change in Gain",:variance_ratio => "L-R DLM Variance Change")*visual(markersize=15,color=:cyan)
f = draw(plot,axis=(; xlabel = "% Gain Change"))
save("FatMothSet1/GainChangeDLMVARChange.png",f,px_per_unit=4)
f
##
dvms = select(powers,:moth,:trial,:ldlmdvm,:rdlmdvm)
leftjoin!(dvms,mean_changes,on=:moth)

plot = data(dvms)*mapping(:mean_fz,:ldlmdvm,color=:trial => 
    renamer(["pre"=>"Lower Mass","post"=>"Higher Mass"]),dodge=:trial => renamer(["pre"=>"pre","post"=>"post"]))*
    visual(BoxPlot,width=0.6)
f = draw(plot,figure=(; resolution=(800,400)), axis=(; xlabel="Relative Change in Fz", ylabel="Left DLM-DVM Timing Offset",title="Left Side"))

save("FatMothSet1/LeftPowerOffsetFZBox.png",f,px_per_unit=4)
f
##


plot = data(dvms)*mapping(:mean_fz,:rdlmdvm,color=:trial => 
    renamer(["pre"=>"Lower Mass","post"=>"Higher Mass"]),dodge=:trial => renamer(["pre"=>"pre","post"=>"post"]))*
    visual(BoxPlot,width=0.6)
f = draw(plot,figure=(; resolution=(800,400)), axis=(; xlabel="Relative Change in Fz", ylabel="Right DLM-DVM Timing Offset",title="Right Side"))

save("FatMothSet1/RightPowerOffsetFZBox.png",f,px_per_unit=4)
f
##
gdf = groupby(dvms, [:moth, :trial])
dvm_means = combine(gdf, :ldlmdvm => mean => :mean_left,:rdlmdvm => mean => :mean_right)
rel_dvm = combine(groupby(dvm_means, :moth)) do group
    (rel_left = group[group.trial .== "post", :mean_left][1] -
                     group[group.trial .== "pre", :mean_left][1],
        rel_right = group[group.trial .== "post", :mean_right][1] -
            group[group.trial .== "pre", :mean_right][1],
                     )
 end
leftjoin!(rel_dvm,unique(select(dvms,:moth,:mean_fz,:mean_gain)),on=[:moth])
##
fig = Figure()
ax = Axis(fig[1,1],
    title="DLM-DVM Timing Offset",
    xlabel = "Gain Change",
    ylabel = "Δ DLM-DVM Timing Offset"
)

l = scatter!(ax,rel_dvm.mean_gain,rel_dvm.rel_left,color=:green,markersize=15)
r = scatter!(ax,rel_dvm.mean_gain,rel_dvm.rel_right,color=:blue,markersize=15)

Legend(fig[1,2],[l,r],["Left Side","Right Side"])

save("FatMothSet1/DVMDLMOffsetGainChange.png",fig,px_per_unit=4)
fig

##

muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
for col in muscle_names
    firstspike[!, col] = convert.(Union{Missing, Float64}, firstspike[!, col])
end

gdf = groupby(firstspike,[:moth,:trial])

meantimes = combine(gdf,
    [name => (x -> var(filter(!ismissing,x))) => name for name in muscle_names]

)
sort!(meantimes, [:moth, :trial])
##
pwise_times = Dict{String, Dict{String, Matrix{Float64}}}()
for moth in moths
    # Initialize nested dictionary for this moth
    pwise_times[moth] = Dict{String, Matrix{Float64}}()
    
    # Get pre and post data for this moth
    moth_data = meantimes[meantimes.moth .== moth, :]
    
    for trial_type in ["pre", "post"]
        # Initialize difference matrix
        diff_matrix = zeros(10, 10)
        
        # Get trial data
        trial_data = moth_data[moth_data.trial .== trial_type, muscle_names]
        
        # Calculate pairwise differences
        for (i, col1) in enumerate(muscle_names)
            for (j, col2) in enumerate(muscle_names)
                if !any(ismissing, trial_data[!, col1]) && !any(ismissing, trial_data[!, col2])
                    diff_matrix[i, j] = mean(trial_data[!, col1]) - mean(trial_data[!, col2])
                else
                    diff_matrix[i, j] = NaN
                end
            end
        end
        
        pwise_times[moth][trial_type] = diff_matrix
    end
end
##

function plot_moth_heatmaps(moth_name, moth_data, muscle_names)
    # Create figure
    fig = Figure(resolution=(1200, 600))
    
    # Find global min/max for consistent color scaling, ignoring NaN values
    all_values = Float64[]
    for matrix in values(moth_data)
        push!(all_values, filter(!isnan, matrix[:])...)
    end
    max_abs_val = maximum(abs.(all_values))
    color_range = (-max_abs_val, max_abs_val)
    
    # Create main title
    fig[1,1:2] = Label(fig, "Pairwise Muscle Differences - $moth_name", fontsize=20)
    
    # Create axes for pre and post
    ax_pre = Axis(fig[2,1], 
                 title="Pre",
                 titlesize=16,
                 aspect=1)
    
    ax_post = Axis(fig[2,2], 
                  title="Post",
                  titlesize=16,
                  aspect=1)
    
    # Create heatmaps
    hm_pre = GLMakie.heatmap!(ax_pre, moth_data["pre"],
                     colormap=:vik, 
                     colorrange=color_range,
                     nan_color=:gray80)
    
    hm_post = GLMakie.heatmap!(ax_post, moth_data["post"],
                      colormap=:vik, 
                      colorrange=color_range,
                      nan_color=:gray80)
    
    # Set ticks and labels
    for ax in [ax_pre, ax_post]
        ax.xticks = (1:length(muscle_names), muscle_names)
        ax.yticks = (1:length(muscle_names), muscle_names)
        ax.xticklabelrotation = π/4
        ax.xticklabelsize = 12
        ax.yticklabelsize = 12
        ax.xlabel = "Muscle"
        ax.ylabel = "Muscle"
    end
    
    # Add colorbar
    Colorbar(fig[2,3], colormap=:vik, limits=color_range,
             flipaxis=false, label="Difference (ms)", width=20,
             labelsize=14)
    
    # Adjust layout
    # colgap!(fig.layout, 20)
    
    return fig
end

##
diff_pwise = Dict(moth => data["post"] - data["pre"] for (moth, data) in pwise_times)

    
# Find global min/max for consistent color scaling, ignoring NaN values
function get_heatmap(moth,dic,muscle_names)
    mat = dic[moth]
    fig = Figure(resolution=(600, 600))
    all_values = Float64[]
    for matrix in values(dic)
        push!(all_values, filter(!isnan, matrix[:])...)
    end
    max_abs_val = maximum(abs.(all_values))
    color_range = (-max_abs_val, max_abs_val)

    # Create main title
    fig[1,1] = Label(fig, "Pairwise Muscle Differences - $moth", fontsize=20)

    # Create axes for pre and post
    ax_pre = Axis(fig[2,1], 
                    aspect=1)


    # Create heatmaps
    hm_pre = GLMakie.heatmap!(ax_pre, mat,
                        colormap=:vik, 
                        colorrange=color_range,
                        nan_color=:gray80)

    # Set ticks and labels
    for ax in [ax_pre]
        ax.xticks = (1:length(muscle_names), muscle_names)
        ax.yticks = (1:length(muscle_names), muscle_names)
        ax.xticklabelrotation = π/4
        ax.xticklabelsize = 12
        ax.yticklabelsize = 12
        ax.xlabel = "Muscle"
        ax.ylabel = "Muscle"
    end

    # Add colorbar
    Colorbar(fig[2,2], colormap=:vik, limits=color_range,
                flipaxis=false, label="Difference (ms)", width=20,
                labelsize=14)

    return(fig)
end
##
for moth in moths 
    f = get_heatmap(moth,diff_pwise,muscle_names)
    save("FatMothSet1/pwisetime/$(moth)diffpwise.png",f,px_per_unit=4)
end
##

for moth in moths
        fig = plot_moth_heatmaps(moth, pwise_times[moth], muscle_names)
        save("FatMothSet1/pwisetime/$(moth).png", fig)
    end
## EHHH heatmaps are hard to parse lets just look at things relative to their own DLM 

time_difs = select(meantimes, 
    :moth,
    :trial,
    [:lax, :ldlm] => ((lax, ldlm) -> lax - ldlm) => :lax,
    [:lba, :ldlm] => ((lba, ldlm) -> lba - ldlm) => :lba,
    [:lsa, :ldlm] => ((lsa, ldlm) -> lsa - ldlm) => :lsa,
    [:ldvm, :ldlm] => ((ldvm, ldlm) -> ldvm - ldlm) => :ldvm,
    [:ldlm, :rdlm] => ((ldlm, rdlm) -> ldlm - rdlm) => :dlm,
    [:rdvm, :rdlm] => ((rdvm, rdlm) -> rdvm - rdlm) => :rdvm,
    [:rsa, :rdlm] => ((rsa, rdlm) -> rsa - rdlm) => :rsa,
    [:rba, :rdlm] => ((rba, rdlm) -> rba - rdlm) => :rba,
    [:rax, :rdlm] => ((rax, rdlm) -> rax - rdlm) => :rax,
    
)

leftjoin!(time_difs,select(mean_changes,:moth,:mean_fz),on=:moth)
##

stacked_times = stack(time_difs,Not(:moth,:trial,:mean_fz))
rename!(stacked_times, :variable => :muscle, :value => :diff)
##

plot = data(stacked_times)*mapping(:mean_fz,:diff,color=:trial => 
    renamer(["pre"=>"Lower Mass","post"=>"Higher Mass"]),layout=:muscle ) * visual(markersize=15)|> draw

##
sorted_df = sort(stacked_times, [:moth, :muscle, :trial])

# Then group by moth and muscle, and calculate difference between post and pre
var_dif = combine(groupby(stacked_times, [:moth,:muscle])) do group
    (vardif = group[group.trial .== "post", :diff][1] -
                     group[group.trial .== "pre", :diff][1],
                     )
end

leftjoin!(var_dif,select(mean_changes,:moth,:mean_fz),on=:moth)
##


plot = data(var_dif)*mapping(:mean_fz,:vardif,layout=:muscle ) * visual(markersize=15,color=:cyan)|> draw
##

sa_difs =  select(meantimes, 
    :moth,
    :trial,
    [:rsa, :lsa] => ((rsa, lsa) -> rsa - lsa) => :sa_off   
)

sa_change = combine(groupby(sa_difs, [:moth])) do group
    (sadiff = group[group.trial .== "post", :sa_off][1] -
                     group[group.trial .== "pre", :sa_off][1],
                     )
end

leftjoin!(sa_change,select(mean_changes,:moth,:mean_fz),on=:moth)
##
sa_change[!, :sadiff] = replace(sa_change[!, :sadiff], NaN => missing)
dropmissing!(sa_change)
##
scatter(sa_change.mean_fz,sa_change.sadiff)