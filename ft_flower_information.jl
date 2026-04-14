using GLMakie, Associations, StatsBase,JLD2, DataFrames,DataFramesMeta,Pipe, FFTW,SavitzkyGolay
using AlgebraOfGraphics
using ProgressMeter
using GLM
include("me_functions.jl")

@load "fat_moths_set_1.jld" allmoths
##
mc = get_mean_changes(allmoths;axis=6)
moths = mc[mc.mean_gain .> 25,:moth]

moth_dict = Dict()
for (i,m) in enumerate(sort(moths))
    moth_dict[m] = "moth_$i"
end
##
function get_mi_lags(subdf;lag=0::Int)
        midf = combine(groupby(subdf,:trial)) do gdf 
            comb = convert(Matrix{Float64},Matrix(select(gdf,:fx,:ty,:tz))) |> StateSpaceSet
            fx = convert(Vector{Float64},gdf.fx) |> StateSpaceSet
            ty = convert(Vector{Float64},gdf.ty) |> StateSpaceSet 
            tz = convert(Vector{Float64},gdf.tz) |> StateSpaceSet 
            
            pos = convert(Vector{Float64},gdf.pos) 
        
            total = information(Shannon(),probabilities(pos))
            (
                fx = association(est,fx[1+lag:end],pos[1:end-lag]),
                ty = association(est,ty[1+lag:end],pos[1:end-lag]),
                tz = association(est,tz[1+lag:end],pos[1:end-lag]),
                all = association(est,comb[1+lag:end],pos[1:end-lag]),
                total = association(est,pos,pos),
            )

        end

    return midf
end

est = KSG2(MIShannon();k=4)

m = moths[1]
all_midf = DataFrame() 
for m in moths
    df = deepcopy(allmoths[m]["data"])
    put_stim_in!(df,allmoths,m)
    subdf = unique(select(df,:wb,:fx,:ty,:tz,:pos,:vel,:trial))
    @showprogress for l in 1:10
        midf = get_mi_lags(subdf;lag=l)
        midf.moth .= moth_dict[m]
        midf.lag .= -1 * l
        all_midf = vcat(all_midf,midf,cols=:union)
    end
end
##
stacke = stack(all_midf,Not(:trial,:total,:moth,:lag),variable_name = "axis",value_name="MI")
plot = data(stacke)*mapping(:lag =>"WS Lag",:MI=>"MI bits/ws",col=:axis,row = :moth,color=:trial)*visual(Lines) |> draw
##
function get_cor_lags(subdf;lag=0::Int)
        midf = combine(groupby(subdf,:trial)) do gdf 
            fx = convert(Vector{Float64},gdf.fx)[1+lag:end]
            ty = convert(Vector{Float64},gdf.ty)[1+lag:end]
            tz = convert(Vector{Float64},gdf.tz)[1+lag:end] 
            pos = convert(Vector{Float64},gdf.pos)[1:end-lag]
            tmp = DataFrame(fx=fx,ty=ty,tz=tz,pos=pos)

            (
                fx = r2(lm(@formula(fx~pos),tmp)),
                ty = r2(lm(@formula(ty~pos),tmp)),
                tz = r2(lm(@formula(tz~pos),tmp))
            )

        end

    return midf
end
##
all_r2 = DataFrame() 
for m in moths
    df = deepcopy(allmoths[m]["data"])
    put_stim_in!(df,allmoths,m)
    subdf = unique(select(df,:wb,:fx,:ty,:tz,:pos,:vel,:trial))
    @showprogress for l in 1:25
        midf = get_cor_lags(subdf;lag=l)
        midf.moth .= moth_dict[m]
        midf.lag .= -1 * l
        all_r2 = vcat(all_r2,midf,cols=:union)
    end
end
##
stacke = stack(all_r2,Not(:trial,:moth,:lag),variable_name = "axis",value_name="r2")
plot = data(stacke)*mapping(:lag =>"WS Lag",:r2=>L"r^2",col=:axis,row = :moth,color=:trial)*visual(Lines) |> draw
