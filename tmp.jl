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
using CategoricalArrays
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("/home/doshna/Documents/PHD/FatMoths/me_functions.jl")
##

##

datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")
moth = "2024_08_01"

moths = ["2024_08_01","2024_08_12","2024_08_14","2024_08_15","2024_08_16"]
##
all_moths = DataFrame()

wb_method = "hilbert"
trial_to_label = Dict(0 => "pre",1=>"mid",2=>"post")
maxwb = 0 
for m in moths
    df = read_ind(datadir,m,wb_method)
    df[!,:trial] = map(i -> trial_to_label[i],df.trial)
    df.wb = df.wb .+ maxwb
    maxwb = maximum(df.wb)
    all_moths = vcat(all_moths,df,cols=:union)
end
save(joinpath(datadir,"allmothscachehilbert.csv"),all_moths)
##

paired_10_colors = [
    colorant"#A6CEE3",  
    colorant"#FB9A99", colorant"#6A3D9A",
    colorant"#1F78B4", colorant"#B2DF8A", 
    colorant"#33A02C", 
     colorant"#E31A1C", 
    colorant"#FDBF6F", colorant"#FF7F00", 
    colorant"#CAB2D6"
]

GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:turquoise,:coral],)
set_theme!(theme)
## lets look at wbfreq 

d4t = @view df[!,:]
trial_to_label = Dict(0 => "pre",1=>"mid",2=>"post")
d4t[!, :trial] = map(i -> trial_to_label[i], d4t.trial)

##
d4t = @view all_moths[all_moths.trial .!= "mid",:]
##
plot = data(d4t)*mapping(:wbfreq => "Wing Beat Frequency",color=:trial =>renamer(["pre" => "Before Feeding", "post" => "After Feeding"])=>"Trial")*histogram(bins=50,normalization=:probability)*visual(alpha=0.7) 
fig = draw(plot,axis=(; ylabel = " "))
save("mungedfigs/freq_with_hilbert.png",fig,px_per_unit=3)
fig
## lets look at muscle phase

plot=data(d4t)*mapping(:phase => "Muscle Phase",row=:muscle,color=:trial=>renamer(["pre" => "Before Feeding", "post" => "After Feeding"])=>"Trial")*AlgebraOfGraphics.histogram(bins=500,normalization=:probability)*visual(alpha=0.5)
fig = draw(plot,figure = (; resolution = (500,800)),axis=(; ylabel=" ",yticks=[0,0.1],))
save("mungedfigs/musc_with_hilbert.png",fig,px_per_unit=3)
fig

##

powers = ["rdlm", "ldlm", "ldvm", "rdvm"]

power_df = all_moths[in.(all_moths.muscle, Ref(powers)),:]

select!(power_df,:wb,:moth,:trial,:muscle,:phase)

##
grouped = groupby(power_df,[:wb,:moth,:trial])

get_min_phase(group, muscle) = minimum(filter(row -> row.muscle == muscle, group).phase)

complete_groups = filter(grouped) do group
    all(muscle in unique(group.muscle) for muscle in powers)
end

result = combine(complete_groups) do group
    rdvm_phase = get_min_phase(group, "rdvm")
    rdlm_phase = get_min_phase(group, "rdlm")
    ldvm_phase = get_min_phase(group, "ldvm")
    ldlm_phase = get_min_phase(group, "ldlm")
    
    (
        Right = rdvm_phase - rdlm_phase,
        Left = ldvm_phase - ldlm_phase
    )
end
##

power_offset_final = stack(result, [:Right, :Left], variable_name = :side, value_name = :diff)

##

d4t = power_offset_final[power_offset_final.trial .!= "mid",:]

plot = data(d4t)*mapping(:trial => renamer(["pre"=>" ","post"=>" "])=> " ",:diff => "Phase Offset",color=:trial => renamer(["pre" => "Before Feeding", "post" => "After Feeding"])=>"Trial",layout=:side)*visual(BoxPlot) 
f = draw(plot,axis=(;))
save("mungedfigs/powertiming.png",f,px_per_unit=4)
f