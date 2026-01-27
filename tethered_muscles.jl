using JLD2
using CSV 
using DataFrames
using DataFrames
using StatsBase
using DataFramesMeta
using CairoMakie
using GLMakie
using Pipe 
using DSP 
using FFTW
using SavitzkyGolay
using AlgebraOfGraphics
using MultivariateStats
include("me_functions.jl")
GLMakie.activate!()
theme = theme_minimal()
theme.fontsize = 20
theme.palette = (; color = [:steelblue,:firebrick])
set_theme!(theme)
@load "fat_moths_set_1.jld" allmoths
fs = Int(1e4)
N = Int(1e5)
##
mc = get_mean_changes(allmoths; axis=6)
moths = mc[mc.mean_gain .> 25,:moth]
m_dict = Dict() 
for (i,m) in enumerate(moths)
    m_dict[m] = Dict()
    m_dict[m]["mass"] = mc[mc.moth .== m,:mass][1]
    m_dict[m]["name"] = "moth_$(i)"
end

all_data = DataFrame() 
for m in moths 
    d = allmoths[m]["data"]
    put_stim_in!(d,allmoths,m)
    all_data = vcat(all_data,d,cols=:union)
end
## No Change to WBF 
wbfreq = unique(select(all_data,:moth,:trial,:wbfreq,:wb))
plot = data(wbfreq)*mapping(:wbfreq,color=:trial => 
    renamer(["pre"=> "Low Mass","post"=> "High Mass"])=> "Body Condition")*
    AlgebraOfGraphics.histogram(bins=100,normalization=:probability)*
    visual(alpha=0.7)|>draw
save("Figs/Final_Teth/wbf.png",plot)
## 
zforce = unique(select(all_data,:moth,:trial,:fz,:wb))
zforce.fz = zforce.fz .* -1000 ## to mN 
zforce.moth .= [m_dict[m]["name"] for m in zforce.moth]
plot = data(zforce)*mapping(:fz => "Z Force (mN)",row = :moth,color=:trial => 
    renamer(["pre"=> "Low Mass","post"=> "High Mass"])=> "Body Condition")*
    AlgebraOfGraphics.histogram(bins=100,normalization=:probability)*
    visual(alpha=0.7)

f = draw(plot,figure=(; size=(1000,1000)),axis=(; limits=(-40,40,nothing,nothing)))

##
d = combine(groupby(all_data,[:moth,:wb,:muscle])) do gdf 
    (
    trial = gdf.trial[1] == "pre" ? 0 : 1, 
    firsttime= minimum(gdf.time),
    firstphase = minimum(gdf.phase),
    count = length(gdf.time),
    wblen = mean(gdf.wblen),
    tz = mean(gdf.tz),
    fz = mean(gdf.fz),
    stim = mean(gdf.pos)
    )
end 

wide_d = unstack(select(d,Not(:count,:firstphase)),:muscle,:firsttime)

dlm = select(wide_d,:moth,:wb,:trial,:wblen,:tz,:stim,:ldlm,:rdlm)
dlm = dropmissing(dlm)
dlm.off = dlm.rdlm .- dlm.ldlm
dlm.trial = [i == 0 ? "pre" : "post" for i in dlm.trial]
dlm.stim = dlm.stim .* 0.14 ## mm/pixels
plot = data(dlm)*mapping(:wb,:tz,color=:trial => 
    renamer(["pre"=> "Low Mass","post"=> "High Mass"])=> "Body Condition",row=:moth)*
    visual(markersize=10,alpha=0.7)|>draw
##
meanvar = combine(groupby(dlm,[:moth,:trial])) do gdf
    (
    mean_off = mean(gdf.off),
    var_off = var(gdf.off),
       )
end
plot = data(meanvar)*mapping(:trial,:var_off,color=:trial => 
    renamer(["pre"=> "Low Mass","post"=> "High Mass"])=> "Body Condition") |> draw
##
dlm = transform(groupby(dlm, [:moth, :trial]), :wb => (x -> x .- minimum(x)) => :new_wb)

plot = data(dlm)*mapping(:new_wb,:off,color=:trial => 
    renamer(["pre"=> "Low Mass","post"=> "High Mass"])=> "Body Condition",row=:moth)*
    visual(markersize=10,alpha=0.7)|>draw
##
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rax","rba","rsa","rdvm","rdlm"]
tmp = dropmissing(wide_d[wide_d.moth.=="2024_11_08",:])
tmp = tmp[tmp.trial.==0,:]
musmat = Matrix(select(tmp,muscle_names))
stim = tmp.stim .* 0.14 ## mm/pixels

mo = fit(MultivariateStats.CCA,musmat',stim')
lat = mo.xproj' * musmat'
# postdf=DataFrame(mus=muscle_names,proj=mo.xproj[:])
##
postdf.proj = postdf.proj ./ sum(abs.(postdf.proj))

predf=  DataFrame(mus = muscle_names,proj=mo.xproj[:])
predf.proj = predf.proj ./ sum(abs.(predf.proj))
predf.trial .= "Low Mass"
postdf.trial .= "High Mass"
tmpdf = vcat(predf,postdf)
tmpdf.proj = abs.(tmpdf.proj)
plot = data(tmpdf)*mapping(:mus,:proj,color=:trial,dodge=:trial)*
    visual(BarPlot)|>draw
##

sub = dropmissing(select(wide_d,[:moth,:wb,:trial,:wblen,:tz,:fz,:lsa,:ldlm]))
sub.loff = sub.lsa .- sub.ldlm 
# sub.roff = sub.rsa .- sub.rdlm
sub = transform(groupby(sub, [:moth,:trial]), :fz => (x -> zscore(x)) => :normfz)

plot = data(sub)*mapping(:loff,:normfz,row=:moth,col=:trial) |> draw