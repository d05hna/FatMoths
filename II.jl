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
using MultivariateStats
using Associations
using Combinatorics
using ProgressMeter
using Base.Threads
using Base.Iterators
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

@load "/home/doshna/Documents/PHD/FatMoths/fat_moths_set_1.jld" allmoths
fs = 1e4 
##
function get_II(moth,allmoths,muscle_names)
    d = allmoths[moth]["data"]
    subs = select(d,:trial,:wb,:muscle,:phase,:fx_pc1,:fx_pc2,:fx_pc3)
    """
    Okay this is really weird but basically the PCA code is returning the pc score and also the pc scre for the prior wb 
    Basically wb 3 will have -8.741,9.842 and then wb 4 will have 6.432,-8.741
    I cannot for the life of me figure out why but I know it always does it this way so I can work around it
    
    """
    firstspike = @pipe d |> 
        groupby(_,[:wb,:muscle]) |> 
        combine(_,:phase=>minimum=>:phase) |>
        unstack(_,:wb,:muscle,:phase)

    fxt = combine(groupby(subs, :wb)) do group
        DataFrame(
            trial = group.trial[1],
            fx_pc1 = group.fx_pc1[1], 
            fx_pc2 = group.fx_pc2[1], 
            fx_pc3 = group.fx_pc3[1] 
        )
        end
    leftjoin!(firstspike,fxt,on=:wb)
    dropmissing!(firstspike)
    pairs = collect(combinations(muscle_names,2))

    data = DataFrame()
    for t in ["pre","post"]
        fxs = convert(Matrix{Float64},Matrix(firstspike[firstspike.trial.==t,["fx_pc1","fx_pc2","fx_pc3"]])) |> StateSpaceSet
        tmp = Dict{Any,Any}()
        tmp["moth"] = moth
        tmp["trial"] = t
        for (m1,m2) in pairs
            if m1 ∉ names(firstspike) || m2 ∉ names(firstspike)
                tmp["$(m1)$(m2)"] = missing
            else
                mus1 = firstspike[firstspike.trial.==t,m1] |> StateSpaceSet
                mus2 = firstspike[firstspike.trial.==t,m2] |> StateSpaceSet
                bofa = convert(Matrix{Float64},Matrix(firstspike[firstspike.trial.==t,[m1,m2]])) |> StateSpaceSet

                Itot = association(KSG2(MIShannon(base=2)),bofa,fxs)
                I1 = association(KSG2(MIShannon(base=2)),mus1,fxs)
                I2 = association(KSG2(MIShannon(base=2)),mus2,fxs)

                II = Itot - I1 - I2 

                tmp["$(m1)$(m2)"] = II 
            end
        end
        push!(data,tmp,cols=:union)
    end

    ##

    nomiss = data[!,[!any(ismissing.(data[!,x])) for x in collect(names(data))]]
    std = stack(nomiss,Not(:moth,:trial),variable_name = "pair",value_name="II")

    diff = combine(groupby(std,[:moth,:pair])) do gdf 
        pr = gdf[gdf.trial.=="pre",:II][1]
        po = gdf[gdf.trial.=="post",:II][1]

        (
            diff = po - pr,
        )
    end

    return(diff)
end
##

moth = "2024_11_04"
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]

mc = get_mean_changes(allmoths)
## 
moths = collect(keys(allmoths))
ks = 1:10 
trials = ["pre","post"]
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
## Loop over every moth trial muscle and k 1:10 to see how stable the estimates are  
itr = Iterators.product(moths,ks,trials,muscle_names)
chunk_size = chunk_size = max(1, length(itr) ÷ nthreads())
data_chunks = partition(itrs,chunk_size)

dfmi = DataFrame()
p = Progress(length(itr); barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',), barlen=20)

tasks = map(data_chunks) do chunk
    @spawn begin
        tmpdf = DataFrame()
        for (moth, k, t, m) in chunk
            next!(p)
            d = allmoths[moth]["data"]
            subs = select(d,:trial,:wb,:muscle,:phase,:fx_pc1,:fx_pc2,:fx_pc3)
            firstspike = @pipe d |> 
                groupby(_,[:wb,:muscle]) |> 
                combine(_,:phase=>minimum=>:phase) |>
                unstack(_,:wb,:muscle,:phase)
        
            fxt = combine(groupby(subs, :wb)) do group
                DataFrame(
                    trial = group.trial[1],
                    fx_pc1 = group.fx_pc1[1], 
                    fx_pc2 = group.fx_pc2[1], 
                    fx_pc3 = group.fx_pc3[1] 
                )
                end
            leftjoin!(firstspike,fxt,on=:wb)
            dropmissing!(firstspike)
            fxs = convert(Matrix{Float64},Matrix(firstspike[firstspike.trial.==t,["fx_pc1","fx_pc2","fx_pc3"]])) |> StateSpaceSet
            tmp = Dict{Any,Any}()
            tmp["moth"] = moth
            tmp["trial"] = t
            tmp["muscle"] = m
            tmp["k"] = k 
            if m ∉ names(firstspike) 
                tmp["MI"] = missing
            else
                mus = firstspike[firstspike.trial.==t,m] |> StateSpaceSet
                mi = association(KSG2(MIShannon(base=2),k=k),mus,fxs)
                tmp["MI"] = mi 
            end

            push!(tmpdf,tmp,cols=:union)
        end
        return tmpdf
    end
end
dfmi = vcat(dfmi, reduce(vcat, fetch.(tasks)))
finish!(p)
##
plot = data(dropmissing(dfmi))*mapping(:k,:MI,color=:trial,row=:muscle,col=:moth)*visual(Lines) |> draw 


##
plot = data(dropmissing(dfmi))*mapping(:MI,row=:muscle,col=:moth,color=:trial)*histogram(bins=50,normalization=:probability)*visual(alpha=0.5)|>draw

## Let me look at the correlations between First Phase and the mean side slip force 
itr = Iterators.product(moths,muscle_names,trials)
chunk_size = chunk_size = max(1, length(itr) ÷ nthreads())
data_chunks = partition(itr,chunk_size)
##
fdf = DataFrame()
tasks = map(data_chunks) do chunk 
    @spawn begin
        tmpdf = DataFrame()
        for (moth, m, t) in chunk
            d = allmoths[moth]["data"]
            subs = select(d,:trial,:wb,:muscle,:phase,:fx)
            firstspike = @pipe d |> 
                groupby(_,[:wb,:muscle]) |> 
                combine(_,:phase=>minimum=>:phase) |>
                unstack(_,:wb,:muscle,:phase)
        
            fxt = combine(groupby(subs, :wb)) do group
                DataFrame(
                    trial = group.trial[1],
                    fx = group.fx[1]
                )
                end
            leftjoin!(firstspike,fxt,on=:wb)
            dropmissing!(firstspike)
            if m in names(firstspike)
                dftmp = firstspike[firstspike.trial.==t,["fx",m]]
                c = abs.(cor(dftmp[!,m],dftmp.fx))
            else 
                c = missing
            end

            tmp = Dict{Any,Any}()
            tmp["moth"] = moth
            tmp["trial"] = t
            tmp["muscle"] = m

            tmp["cor"] = c 

            push!(tmpdf,tmp,cols=:union)

        end
        return(tmpdf)
    end
end

fdf = vcat(fdf, reduce(vcat, fetch.(tasks)))
##
dropmissing!(fdf)

leftjoin!(fdf,mc,on=:moth)
##
plot = data(fdf)*mapping(:mean_gain,:cor,layout=:muscle,color=:trial) |> draw
##
cordif = combine(groupby(fdf,[:moth,:muscle])) do gdf 
    pr = gdf[gdf.trial.=="pre",:cor][1]
    po = gdf[gdf.trial.=="post",:cor][1]

    (dif = po-pr,
    )
end
leftjoin!(cordif,mc,on=:moth)
##
plot = data(cordif)*mapping(:mean_gain,:dif,layout=:muscle)*visual(color=:green) |> draw
## Do the Same for Count 

cdf = DataFrame()
tasks = map(data_chunks) do chunk 
    @spawn begin
        tmpdf = DataFrame()
        for (moth, m, t) in chunk
            d = allmoths[moth]["data"]
            subs = select(d,:trial,:wb,:muscle,:phase,:fx)
            firstspike = @pipe d |> 
                groupby(_,[:wb,:muscle]) |> 
                combine(_,:phase=>length=>:count) |>
                unstack(_,:wb,:muscle,:count)
        
            fxt = combine(groupby(subs, :wb)) do group
                DataFrame(
                    trial = group.trial[1],
                    fx = group.fx[1]
                )
                end
            leftjoin!(firstspike,fxt,on=:wb)
            dropmissing!(firstspike)
            if m in names(firstspike)
                dftmp = firstspike[firstspike.trial.==t,["fx",m]]
                c = abs.(cor(dftmp[!,m],dftmp.fx))
            else 
                c = missing
            end

            tmp = Dict{Any,Any}()
            tmp["moth"] = moth
            tmp["trial"] = t
            tmp["muscle"] = m

            tmp["cor"] = c 

            push!(tmpdf,tmp,cols=:union)

        end
        return(tmpdf)
    end
end
cdf = vcat(cdf, reduce(vcat, fetch.(tasks)))

##
dropmissing!(cdf)
leftjoin!(cdf,mc,on=:moth)
##
plot = data(cdf)*mapping(:mean_gain,:cor,layout=:muscle,color=:trial) |> draw 
## 
"""
So Basic Count and phase have no correlations with The FX and those dont change at all cool
What else can we do to make this work in a simple linear way? 
""" 

d = allmoths["2025_01_30"]["data"]
subs = select(d,:trial,:wb,:muscle,:phase,:fx)
subs.fx = round.(subs.fx,digits=4)
firstspike = @pipe d |> 
    groupby(_,[:wb,:muscle]) |> 
    combine(_,:phase=>minimum=>:phase) |>
    unstack(_,:wb,:muscle,:phase)
##
dropmissing!(firstspike)

wbtfx = combine(groupby(subs,[:wb,:trial])) do gdf
    (fx = gdf.fx[1],)
    end

leftjoin!(firstspike,wbtfx,on=:wb)
df = firstspike[!,Not(:wb,:trial,:fx)]

phasemat = Matrix(df[!,:])'


##
res = MultivariateStats.fit(MultivariateStats.PCA,phasemat;maxoutdim=3)
latent = MultivariateStats.predict(res,phasemat)'

lat = DataFrame(Matrix(latent),["pc1","pc2","pc3"])
lat.trial = firstspike.trial
lat.wb = firstspike.wb
lat.fx = firstspike.fx

plot = data(lat)*mapping(:pc1,:pc2,:pc3,color=:trial)
draw(plot, axis=((type=Axis3,)))
##

ld = MultivariateStats.fit(MulticlassLDA,Matrix(firstspike[!,Not(:wb,:trial,:fx)])',firstspike.trial;outdim=2)

ou = MultivariateStats.predict(ld,Matrix(firstspike[!,Not(:wb,:trial,:fx)])')

fig = Figure()
ax = Axis(fig[1,1])
for t in ["pre","post"]
    po = ou[:,firstspike.trial.==t]
    scatter!(ax,po[1,:],po[2,:],label=t)
end
Legend(fig[1,2],ax)
fig
##
we = DataFrame(ld.proj,["ld1","ld2"])
we.muscle = muscle_names[1:end-1]
we = stack(we,[:ld1,:ld2])

plot = data(we)*mapping(:muscle,:value,color=:variable) |>draw
##

subs = select(d,:trial,:wb,:muscle,:phase,:fx_pc1,:fx_pc2,:fx_pc3)

firstspike = @pipe d |> 
    groupby(_,[:wb,:muscle]) |> 
    combine(_,:phase=>minimum=>:phase) |>
    unstack(_,:wb,:muscle,:phase)

fxt = combine(groupby(subs, :wb)) do group
    DataFrame(
        trial = group.trial[1],
        fx_pc1 = group.fx_pc1[1], 
        fx_pc2 = group.fx_pc2[1], 
        fx_pc3 = group.fx_pc3[1] 
    )
    end
leftjoin!(firstspike,fxt,on=:wb)
dropmissing!(firstspike)
##

spikesmat = Matrix(firstspike[!,muscle_names[1:end-1]])'

pcmat = Matrix(firstspike[!,["fx_pc$i" for i in 1:3]])'

cc = MultivariateStats.fit(CCA,spikesmat,pcmat)

projs = DataFrame(abs.(projection(cc,:x)),["cv1","cv2","cv3"])
projs.muscle = muscle_names[1:end-1]
lp = stack(projs,[:cv1,:cv2,:cv3],variable_name = "cv",value_name="weight")

plot = data(lp)*mapping(:muscle,:weight,color=:cv,dodge=:cv)*visual(BarPlot) |>draw
##
lat = DataFrame(MultivariateStats.predict(cc,spikesmat,:x)',["cv1","cv2","cv3"])
lat.trial = firstspike.trial

fig = Figure() 
ax = Axis3(fig[1,1],xlabel="CV1",ylabel="CV2",zlabel="CV3")

for t in ["pre","post"]
    tmp = lat[lat.trial.==t,:]
    scatter!(ax,tmp.cv1,tmp.cv2,tmp.cv3,label=t)
end
Legend(fig[1,2],ax)

start_angle = π / 4
n_frames = 120
ax.viewmode = :fit # Prevent axis from resizing during animation
record(fig, "test.gif", 1:n_frames) do frame
    ax.azimuth[] = start_angle + 2pi * frame / n_frames
end