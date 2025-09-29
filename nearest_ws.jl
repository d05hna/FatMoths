using DataFrames 
using CSV
using LinearAlgebra
using JLD2
using AlgebraOfGraphics
using GLMakie
using NearestNeighbors
using SavitzkyGolay
using DSP
using FFTW
using StatsBase
using Pipe
using DataFramesMeta
using MultivariateStats
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
# theme.palette = (color=[:steelblue,:firebrick],)
include("me_functions.jl")
set_theme!(theme)
##

@load "/home/doshna/Documents/PHD/FatMoths/fat_moths_set_1.jld" allmoths
##
function nearest_wb_flower(allmoths,moth)
    df = allmoths[moth]["data"]

    spre = allmoths[moth]["stimpre"]
    spost = allmoths[moth]["stimpost"]

    sgpre = savitzky_golay(spre,11,4;deriv=0)
    vpre = savitzky_golay(spre, 11, 4; deriv=1)

    sgpost = savitzky_golay(spost,11,4;deriv=0)
    vpost = savitzky_golay(spost,11,4;deriv=1)


    stimtime = range(0,10,length=length(spre))
    ##
    dfpre = df[df.trial.=="pre",:]
    dfpre.time_abs .-= minimum(dfpre.time_abs)
    dfprewbtime = combine(groupby(dfpre,:wb),
        :time_abs => mean => :wbtime
    )
    prestimidx = [findlast(x-> x  <= i,stimtime) for i in dfprewbtime.wbtime]
    dfprewbtime.pos = sgpre.y[prestimidx]
    dfprewbtime.vel = vpre.y[prestimidx]

    dfpost = df[df.trial.=="post",:]
    dfpost.time_abs .-= minimum(dfpost.time_abs)
    dfpostwbtime = combine(groupby(dfpost,:wb),
        :time_abs => mean => :wbtime
    )
    poststimidx = [findlast(x-> x  < i,stimtime) for i in dfpostwbtime.wbtime]
    dfpostwbtime.pos = sgpost.y[poststimidx]
    dfpostwbtime.vel = vpost.y[poststimidx]
    ##

    wbpre = unique(dfpre.wb)
    wbpo = unique(dfpost.wb)
    matpo = Matrix(dfpostwbtime[!,[:pos,:vel]])'
    matpre = Matrix(dfprewbtime[!,[:pos,:vel]])'

    # comb = hcat(matpo,tft)

    tre = KDTree(matpo)
    k=4
    idx, di = knn(tre,matpre,k)
    ##
    sorted_indices = [inds[sortperm(dists)] for (inds, dists) in zip(idx, di)]

    indicies = DataFrame("wbpre"=>wbpre,"idxs"=>sorted_indices)
    indicies.k1 = [wbpo[indicies.idxs[i][1]] for i in 1: nrow(indicies)]
    indicies.k2 = [wbpo[indicies.idxs[i][2]] for i in 1: nrow(indicies)]
    indicies.k3 = [wbpo[indicies.idxs[i][3]] for i in 1: nrow(indicies)]
    indicies.k4 = [wbpo[indicies.idxs[i][4]] for i in 1: nrow(indicies)]

    ##
    indicies.wbidx = [Matrix(indicies[!,[:k1,:k2,:k3,:k4]])[i,:] for i in 1:nrow(indicies)]
    rename!(indicies, [:wbpre => :wb])
    select!(indicies,:wb,:wbidx)
    ##
    firstspike = @pipe df |> 
        groupby(_,[:wb,:muscle]) |> 
        combine(_,:time=>minimum=>:time) |>
        unstack(_,:wb,:muscle,:time)
    dropmissing!(firstspike)
    firstspike.trial = [x ∈ wbpre ? "pre" : "post" for x in firstspike.wb]
    ##
    muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]

    comps = DataFrame()
    for wb in firstspike[firstspike.trial.=="pre",:wb]
        twb = firstspike[firstspike.wb.==wb,:]
        id = indicies[indicies.wb.==wb,:wbidx][1]
        sub = firstspike[in.(firstspike.wb,(id,)),:]
        tmp = Dict()
        tmp["wb"] = wb
        for m in muscle_names
            if m in names(twb) && length(sub[!,m])>0
                tmp[m] = mean(sub[!,m] .- twb[!,m])
            else
                tmp[m] = missing
            end
        end

        tmp["pos"] = dfprewbtime[dfprewbtime.wb.==wb,:pos][1]
        tmp["vel"] = dfprewbtime[dfprewbtime.wb.==wb,:vel][1] 
        push!(comps,tmp,cols=:union)
    end

    comps = stack(comps,muscle_names,variable_name="muscle",value_name="timedif")
    comps.pos .-= mean(comps.pos)
    dropmissing!(comps)
    ##

    plot = data(comps)*mapping(:pos => "Flower Position (Pixel)",:timedif => "First Spike Timing Difference",layout=:muscle)*visual(color=:steelblue)
    f = draw(plot,figure=(; size=(800,800)))
    save("nnfigs/Pos/$(moth).png",f)
    f
    ##

    plot = data(comps)*mapping(:vel => "Flower Velocity (Pixel/Frame)",:timedif => "First Spike Timing Difference",layout=:muscle)*visual(color=:steelblue)
    f = draw(plot,figure=(; size=(800,800)))
    save("nnfigs/Vel/$(moth).png",f)
    f
    ##
    return comps
end
##
fs = 1e4
stimless = []
##
mc = get_mean_changes(allmoths)
allc = DataFrame()
for m in keys(allmoths)
    if m in stimless
        println("This Is a Moth I haven't Dont The Stim for Yet: $m")
    else
        println("Doing Moth: $m")
        c = nearest_wb_flower(allmoths,m)
        c.moth .= m
        allc = vcat(allc,c,cols=:union)
    end
end
##


##
df = allmoths["2024_11_08"]["data"]
put_stim_in!(df,allmoths,"2024_11_08")
##
function get_means_by_fx(moth,allmoths)
    df = allmoths[moth]["data"]
    if !("pos" in names(df))
        put_stim_in!(df,allmoths,moth)
    end
    df.pos = convert(Vector{Float64},df.pos)


    tmp = unique(select(df,:wb,:trial,:moth,:fx))
    tmp = combine(groupby(tmp,:wb)) do gdf 
        (
            moth = gdf.moth[1],
            fx = gdf.fx[1],
            trial = gdf.trial[1],
        )
    end

    firstspike = @pipe df |> 
        groupby(_,[:wb,:muscle]) |> 
        combine(_,:time=>minimum=>:time) |>
        unstack(_,:wb,:muscle,:time)
    dropmissing!(firstspike)
    
    leftjoin!(firstspike,tmp,on=:wb)

    shortmus = ["ax","ba","sa","dvm","dlm"]
    used = []
    for m in shortmus
        if ("l"*m in names(firstspike)) && ("r"*m in names(firstspike))
            firstspike[!,m] = firstspike[!,"r"*m] - firstspike[!,"l"*m]
            push!(used,m)
        end
    end
    firstspike = add_relfx_column!(firstspike)
    offsets = select(firstspike,"trial",used,:relfx)
    # offsets = stack(offsets,Not(:trial,:relfx))
    # offsets = offsets[offsets.value .< 0.01,:]/
    # offsets = unstack(offsets,combine=mean]
    result = combine(
            groupby(offsets, [:trial, :relfx]),
            [Symbol(col) => (x -> var(skipmissing(x))) => Symbol(col) for col in used]...
        )
    vars = stack(result,Not(:trial,:relfx),variable_name = "mus",value_name="meanoff")

    return(vars)
end
##
function get_means_by_fx_ref(moth,allmoths;ref="dlm")
    df = allmoths[moth]["data"]
    if !("pos" in names(df))
        put_stim_in!(df,allmoths,moth)
    end
    df.pos = convert(Vector{Float64},df.pos)


    tmp = unique(select(df,:wb,:trial,:moth,:fx))
    tmp = combine(groupby(tmp,:wb)) do gdf 
        (
            moth = gdf.moth[1],
            fx = gdf.fx[1],
            trial = gdf.trial[1],
        )
    end

    firstspike = @pipe df |> 
        groupby(_,[:wb,:muscle]) |> 
        combine(_,:time=>minimum=>:time) |>
        unstack(_,:wb,:muscle,:time)
    dropmissing!(firstspike)
    
    leftjoin!(firstspike,tmp,on=:wb)

    shortmus = ["ax","ba","sa","dvm","dlm"]
    used = []
    for m in shortmus
        if ("l"*m in names(firstspike)) 
            firstspike[!,"l"*m] = firstspike[!,"l"*m] - firstspike[!,"l"*ref]
            push!(used,"l"*m)
        end
        if ("r"*m in names(firstspike))
            firstspike[!,"r"*m] = firstspike[!,"r"*m] - firstspike[!,"r"*ref]
            push!(used,"r"*m)
        end
    end
    firstspike = add_relfx_column!(firstspike)
    offsets = select(firstspike,"trial",used,:relfx)
    # offsets = stack(offsets,Not(:trial,:relfx))
    # offsets = offsets[offsets.value .< 0.01,:]
    # offsets = unstack(offsets,combine=mean)
    result = combine(
            groupby(offsets, [:trial, :relfx]),
            [Symbol(col) => (x -> mean(skipmissing(x))) => Symbol(col) for col in used]...
        )
    vars = stack(result,Not(:trial,:relfx),variable_name = "mus",value_name="meanoff")

    return(vars)
end

##
meanoffs = DataFrame()
for moth in collect(keys(allmoths))
    mo = get_means_by_fx(moth,allmoths)
    mo.moth .=moth
    meanoffs = vcat(meanoffs,mo,cols=:union)
end
##
meanoffs_ref = DataFrame()
for moth in collect(keys(allmoths))
    mo = get_means_by_fx_ref(moth,allmoths)
    mo.moth .=moth
    meanoffs_ref = vcat(meanoffs_ref,mo,cols=:union)
end
##
mc = get_mean_changes(allmoths)

leftjoin!(meanoffs,mc,on=:moth)
##
# mor = meanoffsf[meanoffs_ref.mus .!= "ldlm" .&& meanoffs_ref.mus .!= "rdlm",:]
mor = meanoffs[meanoffs.moth.=="2024_11_08".||meanoffs.moth.=="2024_11_01",:]
# mor.meanoff = abs.(mor.meanoff)
##
plot = data(mor)*mapping(:relfx => "Binned Side Slip Force",:meanoff => "Var Timing Offset From LR (s)",
    color=:trial => renamer(["pre"=> "Low Mass","post"=>"High Mass"]),col=:moth,row=:mus) *
    visual(markersize=15)
f = draw(plot,axis=(; xlabelfont=:bold,ylabelfont=:bold))

##

plot = data(meanoffs[meanoffs.mus.=="ax",:])*mapping(:relfx,:meanoff,color=:trial,layout=:moth) |> draw

##
function get_cca_cor(allmoths,moth,trial)
    df = allmoths[moth]["data"]
    tmp = unique(select(df,:wb,:trial,:moth,:pos,:vel))
    tmp = combine(groupby(tmp,:wb)) do gdf 
        (
            moth = gdf.moth[1],
            pos = gdf.pos[1],
            trial = gdf.trial[1],
            vel = gdf.vel[1],
        )
    end

    firstspike = @pipe df |> 
        groupby(_,[:wb,:muscle]) |> 
        combine(_,:time=>minimum=>:time) |>
        unstack(_,:wb,:muscle,:time)
    dropmissing!(firstspike)

    leftjoin!(firstspike,tmp,on=:wb)
    firstspike.pos = convert(Vector{Float64},firstspike.pos)
    firstspike.vel = convert(Vector{Float64},firstspike.vel)

    ##

    pr = firstspike[firstspike.trial.==trial,:]

    ma = Matrix(select(pr,Not(:wb,:moth,:trial,:pos,:vel)))'

    fpr = Matrix(select(pr,:pos,:vel))'
    ##
    M = fit(CCA,ma,fpr)
    return M.corrs
end

##
Out= DataFrame()
for moth in collect(keys(allmoths))
    for t in ["pre","post"]
        c = get_cca_cor(allmoths,moth,t)
        Out= push!(Out,Dict("moth"=>moth,"trial"=>t,"cor"=>c[1]),cols=:union)
    end
end
##
plot = data(Out)*mapping(:trial,:cor,color=:moth)
f = draw(plot,axis=(; limits=(nothing,nothing,0,1),xreversed=true))