big_lax_dlm = CSV.read("workingcopy.csv",DataFrame)
##
moth = "2024_11_05"
d = allmoths[moth]["data"]
##

res = analyze_muscle_timing(d,"ldlm","rdlm")
dropmissing!(res)
##
lines(res[res.trial.=="pre",:time_difference])
##
pre = res[res.trial.=="pre",:]
# pre = pre[pre.time_difference .> 0.01 .&& pre.time_difference .< 1,:]
pre.time = pre.wstime .- minimum(pre.wstime)
lines(pre.time_difference)
##
pos = res[res.trial.=="post",:]
pos = pos[pos.time_difference .>  0.0001 .&& pos.time_difference .< 0.004,:]
pos.time = pos.wstime .- minimum(pos.wstime)
lines(pos.time_difference)
##
tmp = DataFrame(
    "time" => pre.time,
    "diff" => pre.time_difference
)
tmp.trial .= "pre"
tmp.moth .= moth 
tmp.mus .= "dlm"

big_lax_dlm = vcat(big_lax_dlm,tmp,cols=:union)
##
tmp = DataFrame(
    "time" => pos.time,
    "diff" => pos.time_difference
)
tmp.trial .="post"
tmp.moth .= moth
tmp.mus .= "dlm"
big_lax_dlm = vcat(big_lax_dlm,tmp,cols=:union)
##

vars_lax_dlm = combine(groupby(big_lax_dlm,[:moth,:trial,:mus]),
    :diff => var => :variance)
##
diff_vars = combine(groupby(vars_lax_dlm, [:moth,:mus])) do group_df

    pre = filter(row -> row.trial == "pre", group_df)
    post = filter(row -> row.trial == "post", group_df)
    
    # Calculate min time difference
    (
        vardif = 100* (post.variance[1] - pre.variance[1])/pre.variance[1] ,
    )
end
##
leftjoin!(diff_vars,mean_changes,on=:moth)
##

plot = data(diff_vars)*mapping(:mean_fz,:vardif,col=:mus)*visual(markersize=20,color=:green) |> 

##
biglax = analyze_muscle_timing(all_data,"lax","ldlm")
dropmissing!(biglax)
rename!(biglax, ["m1count" => "laxcount","m2count"=>"ldlmcount"])
##
cors = combine(groupby(biglax,[:moth,:trial])) do gdf 
    (
        countcor = cor(gdf.laxcount,gdf.tz),
        timecor = cor(gdf.time_difference,gdf.tz)
    )
end
##
bigdlm = analyze_muscle_timing(all_data,"ldlm","rdlm")
dropmissing!(bigdlm)
rename!(bigdlm, ["m1count" => "ldlmcount","m2count"=>"rdlmcount"])
##
cccs= combine(groupby(bigdlm,[:moth, :trial])) do gdf 
    (
        cor = maximum(abs.(crosscor(gdf.time_difference,gdf.tz))),
    )
end
##
diffcor = combine(groupby(cccs, [:moth])) do group_df

    pre = filter(row -> row.trial == "pre", group_df)
    post = filter(row -> row.trial == "post", group_df)
    
    # Calculate min time difference
    (
        cordif = 100*(post.cor[1] - pre.cor[1])/pre.cor[1],
    )
end
##
regcor = lm(@formula(cordif~mean_gain),diffcor)
rcor = r2(regcor)
fte = GLM.ftest(regcor.model)

##
x_new = range(-30,100,length=100)
pr = predict(regcor, DataFrame(mean_gain = x_new),interval = :confidence, level = 0.95)

##
f = Figure()
ax = Axis(f[1,1],
    xlabel = "% Change in Controller Gain",
    ylabel = "% Change in cor(dlm,fx)",
    )
scatter!(ax,diffcor.mean_gain,diffcor.cordif,markersize=20)
lines!(ax,x_new,convert(Vector{Float64},pr.prediction),linewidth=4)
text!(ax, 
    "R² = $(round(rcor, digits=3))",
    position = (-20,200),
    fontsize = 25,
    color=:red
)
text!(ax, 
    "p = $(round(fte.pval, digits=3))",
    position = (-19.5,165),
    fontsize = 25,
    color=:red
)
f 
##
time = forfig.wstime .- minimum(forfig.wstime)
fig = Figure(resolution=(800,600))
ax1 = Axis(fig[1,1],ylabel="Δ DLM Timing (ms)",xlabel="Time (s)",ylabelcolor=dlmcolor,yticklabelcolor=dlmcolor)
scatter!(ax1,time[14:end],forfig.time_difference[14:end] .* 1e3,markersize=10,color=dlmcolor)
ax2 = Axis(fig[1,1],ylabel="Side Slip Force (mN)",yaxisposition=:right,ylabelcolor=:steelblue,yticklabelcolor=:steelblue)
scatter!(ax2,time.+0.5,forfig.fx .* 1e3,color=:steelblue)
linkxaxes!(ax1,ax2)
ax1.limits = (0,10,nothing,nothing)

save("SICBFigs/DLMFX.png",fig,px_per_unit=4)
fig
##
f = Figure(resolution=(800,800))
ax = Axis(f[1,1],ylabel="Correlation",limits=(-0.5,1.5,0.,0.8),xticksvisible=false,xticklabelsvisible=false)
scatter!(ax,cccs.tri,cccs.newcor,color=cccs.tri,colormap=[skinnyblue,fatred],markersize=25)

f
save("SICBFigs/DotsCorDLM.png",f,px_per_unit=4)