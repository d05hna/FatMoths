using DataFrames
using CSV 
using StatsBase
using GLMakie 
using AlgebraOfGraphics
using Glob
using Associations
using FFTW
using JLD2
using Pipe
using DataFramesMeta
using SavitzkyGolay
include("me_functions.jl")
##
GLMakie.activate!()
theme = theme_black()
theme.palette = (color = [:steelblue,:firebrick],)
theme.fontsize = 20
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
datadir = "/home/doshna/Documents/PHD/data/fatties/Muscle_Time_Series"
fs = glob("*.csv",datadir)
ftnames = ["fx","fy","fz","tx","ty","tz"]
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

##
Out = DataFrame()
for f in fs
    mof = split(f,"/")[end][1:10]
    bod = split(split(f,"/")[end],"_")[end][1:end-4]

    df = select(CSV.read(f,DataFrame),ftnames,"stim")
    df_scored = deepcopy(df)
    for ft in ftnames
        df_scored[!,ft] = (df_scored[!,ft] .- mean(df_scored[!,ft])) / std(df_scored[!,ft])
    end
    df_scored[!,"stim"] = (df_scored[!,"stim"] .- mean(df_scored[!,"stim"])) / std(df_scored[!,"stim"])
    df_scored.moth.=mof
    df_scored.condition .=bod
    Out = vcat(Out,df_scored,cols=:union)
end
##
stacked = stack(Out,Not(:moth,:condition),variable_name="axis",value_name="score")

##
plot = data(stacked)*mapping(:score,color=:moth,col=:axis,row=:condition)*AlgebraOfGraphics.density() |> draw
##

est = KSG2(MIShannon();k=4)
sub = @view Out[Out.moth.=="2024_11_01".&& Out.condition.=="HIGH",:]
ftmat = sub.stim
stimmat = sub.stim
mi_vals = combine(groupby(Out,[:moth,:condition])) do gdf
    (
        fx = association(est,StateSpaceSet(gdf.fx),StateSpaceSet(gdf.stim)),
        fy = association(est,StateSpaceSet(gdf.fy),StateSpaceSet(gdf.stim)),
        fz = association(est,StateSpaceSet(gdf.fz),StateSpaceSet(gdf.stim)),
        tx = association(est,StateSpaceSet(gdf.tx),StateSpaceSet(gdf.stim)),
        ty = association(est,StateSpaceSet(gdf.ty),StateSpaceSet(gdf.stim)),
        tz = association(est,StateSpaceSet(gdf.tz),StateSpaceSet(gdf.stim))
    )
end
##
mi_stacked = stack(mi_vals,Not(:moth,:condition))
plot = data(mi_stacked)*mapping(:variable => "Axis",:value => "MI(FT,Stim) [bits]",
    color=:condition =>renamer(["LOW"=>"Low Mass","HIGH"=>"High Mass"]),dodge=:condition)*visual(BoxPlot)|>draw

##
tmp = CSV.read(fs[12],DataFrame)

ft_tx = fft(tmp.tx)[2:250]
fr  = fftfreq(length(tmp.tx),10000)[2:250]

f = Figure()
ax = Axis(f[1,1],xscale=log10)
lines!(fr,abs.(ft_tx))
vlines!(freqqs,linestyle=:dash,color=:grey,alpha=0.5)
f
##
@load "fat_moths_set_1.jld" allmoths
##
all = DataFrame()
for moth in collect(keys(allmoths))
    d = allmoths[moth]["data"]
    put_stim_in!(d,allmoths,moth)
    all = vcat(all,d,cols=:union)
end
##
colnames = [x * "_pc$i" for x in ftnames for i in 1:3]
posdf = unique(select(all,"moth","trial","wb","pos",ftnames,colnames))

mi_all_pcs = combine(groupby(posdf,[:moth,:trial])) do gdf 
    fts = Dict()
    for ft in ftnames 
        mat = convert(Matrix{Float64},Matrix(select(gdf,["$(ft)_pc$i" for i in 1:3])))
        posmat = convert(Matrix{Float64},Matrix(select(gdf,"pos")))
        fts[ft] = association(est,StateSpaceSet(mat),StateSpaceSet(posmat))
    end
    (
        fx = fts["fx"],
        fy = fts["fy"],
        fz = fts["fz"],
        tx = fts["tx"],
        ty = fts["ty"],
        tz = fts["tz"],
    )
end
##
mi_stacked = stack(mi_all_pcs,Not(:moth,:trial))
plot = data(mi_stacked)*mapping(:variable => "Axis",:value => "MI(FT,Stim) [bits per ws]",
    color=:trial =>renamer(["pre"=>"Low Mass","post"=>"High Mass"]),dodge=:trial)*visual(BoxPlot)|>draw
