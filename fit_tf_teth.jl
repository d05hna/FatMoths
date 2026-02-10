using DataFrames
using JLD2 
using StatsBase
using FFTW
using SpecialFunctions
using LinearAlgebra
using Pipe 
using CairoMakie
using DataFramesMeta
using Interpolations
using DSP
using GLM
using Optim
using ProgressMeter
include("stats_functions.jl")
include("me_functions.jl")
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moths_set_1.jld" 
fs = Int(1e4)
N = Int(1e5)
freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
pixel_conversion = 0.14 ## mm/pixels 


mc = get_mean_changes(allmoths; axis=6)
moths = mc[mc.mean_gain .> 25,:moth]

Hi_fx = zeros(ComplexF64,18,length(moths))
Hf_fx = zeros(ComplexF64,18,length(moths))
Hi_yaw = zeros(ComplexF64,18,length(moths))
Hf_yaw = zeros(ComplexF64,18,length(moths))
Hi_roll = zeros(ComplexF64,18,length(moths))
Hf_roll = zeros(ComplexF64,18,length(moths))

for (i,m) in enumerate(moths)
    dat = allmoths[m]["ftpre"]
    datf = allmoths[m]["ftpos"]

    fpre = (allmoths[m]["stimpre"] .- mean(allmoths[m]["stimpre"])) .* pixel_conversion
    fpost = (allmoths[m]["stimpost"] .- mean(allmoths[m]["stimpost"])) .* pixel_conversion

    Hi_fx[:,i] = tf_freq(fpre,dat[:,1],freqqs,fs)
    Hf_fx[:,i] = tf_freq(fpost,datf[:,1],freqqs,fs)

    Hi_yaw[:,i] = tf_freq(fpre,dat[:,6],freqqs,fs)
    Hf_yaw[:,i] = tf_freq(fpost,datf[:,6],freqqs,fs)
    Hi_roll[:,i] = tf_freq(fpre,dat[:,5],freqqs,fs)
    Hf_roll[:,i] = tf_freq(fpost,datf[:,5],freqqs,fs)
end
##
function error_model(y_pred,y;gw=100000,pw=1)
    # same thing as the cowan paper, but split up so we can weight 
    e = abs.(log.(y_pred ./ y)) .^2 
    return(sum(e))
end 

function make_model(n_a::Int, n_b::Int, d::Real)
    function f(p::AbstractVector, x::Real)
        # parameter layout:
        # [a0..a_n, b0..b_m, k, T]
        x = im * 2pi * x
        a = @view p[1:n_a]
        b = @view p[n_a+1 : n_a+n_b+1]
        k = p[end-1]
        T = p[end]

        num = sum(a[i+1] * x^i for i in 0:n_a-1)
        den = sum(b[j+1] * x^j for j in 0:n_b-1)

        return k * num / den * exp(-d * x * T)
    end
    return f
end

function eval_model(p,mod_fun,y_true,freqqs)
    y_pred = [mod_fun(p,f) for f in freqqs]
    e = error_model(y_pred,y_true)
    return e
end

function any_given_model(np,nz,d,z,freqqs;return_error = false)
    if return_error
        errors = [] 
    end

    best_e = Inf 
    best_ps = nothing 
    mod_fun = make_model(nz,np,d) 
    for _ in 1:100
        p0 = vcat([rand() for _ in 1:nz],[rand() for _ in 1:np],[rand(),rand()])
        res = optimize(p -> eval_model(p,mod_fun,z,freqqs[1:15]),p0,NelderMead())
        e = Optim.minimum(res)
        if e < best_e
            best_e = e
            best_ps = Optim.minimizer(res)
        end
        if return_error
            push!(errors,e)
        end
    end
    if return_error
        return mod_fun,best_ps,errors
    else
        return mod_fun,best_ps

    end
end

function dropout_all(Mat,np,nz,d;freqqs=freqqs[1:15])
    N = size(Mat,2)
    errors = zeros(N)
    ps = zeros(np+nz+2,N)
    sub = []
    for i in 1:N 
        sub = Mat[:,Not(i)]
        ms,_ = get_all_tf_stats(sub,Hf_yaw,freqqs; freq_max=8)
        mg = ms.mg 
        mp = ms.mp 
        z = mg .* exp.(im .* mp)
        fun,best_ps = any_given_model(np,nz,d,z,freqqs)
        ps[:,i] = best_ps
        dropout_error= eval_model(best_ps,fun,Mat[:,i],freqqs)
        errors[i] = dropout_error
    end
    
    mean_e = mean(errors) 
    for i in 1:np+nz+2
        ps[i,:] .-= mean(ps[i,:])
    end
    svmax = opnorm(ps)
    return(mean_e,svmax)
end

## Lets Start by just looking at low mass Yaw
nps = Int.(range(1,5,5))
nzs = Int.(range(1,5,5))
ds  = [0,1]
sub = []
low_yaw = DataFrame(np=Int[],nz=Int[],d=Int[],mean_error=Float64[],svmax=Float64[]) 
bar = Progress(length(nps)*length(nzs)*length(ds),desc="Fitting Models")
for np in nps 
    for nz in nzs
        for d in ds 
            if nz>np
                next!(bar)
            else
                mean_e,svmax = dropout_all(Hi_yaw[1:15,:],np,nz,d)
                push!(low_yaw,(np=np,nz=nz,d=d,mean_error=mean_e,svmax=svmax))
                next!(bar)
            end
        end
    end
end
##
high_yaw = DataFrame(np=Int[],nz=Int[],d=Int[],mean_error=Float64[],svmax=Float64[]) 
bar = Progress(length(nps)*length(nzs)*length(ds),desc="Fitting Models")
sub = []
for np in nps 
    for nz in nzs
        for d in ds 
            if nz>np
                next!(bar)
            else
                mean_e,svmax = dropout_all(Hf_yaw[1:15,:],np,nz,d)
                push!(high_yaw,(np=np,nz=nz,d=d,mean_error=mean_e,svmax=svmax))
                next!(bar)
            end
        end
    end
end
##
f = Figure(size=(800,800)) 
ax = Axis(f[1,1],
    xlabel = "Max Singular Value of Parameters (a.u.)",
    ylabel = "Mean Dropout Error (a.u.)",
    title = "Low Mass Yaw TF Model ",
    xscale = log10,    limits=(10^-0.5,10^5.8,nothing,nothing))

scatter!(ax,low_yaw.svmax,low_yaw.mean_error,color=low_yaw.d)
scatter!(ax,13.8772,34.4471,color=:transparent,strokecolor=:red,strokewidth=2,markersize=30)
text!(ax,7.47247,37,text="(3,2,1)",align = (:center,:center),color=:red,fontsize=20)
scatter!(ax,7122.19,44.1324,color=:transparent,strokecolor=:blue,strokewidth=2,markersize=30)
text!(ax,7122.19,47,text="(3,1,1)",align = (:center,:center),color=:blue,fontsize=20)
ax2 = Axis(f[2,1],
    xlabel = "Max Singular Value of Parameters (a.u.)",
    ylabel = "Mean Dropout Error (a.u.)",
    title = "High Mass Yaw TF Model ",
    xscale = log10,
    limits=(10^-0.5,10^5.8,nothing,nothing))
scatter!(ax2,high_yaw.svmax,high_yaw.mean_error,color=high_yaw.d)
scatter!(ax2,9833.69,52.9517,color=:transparent,strokecolor=:red,strokewidth=2,markersize=30)
text!(ax2,9833,55,text="(3,2,1)",align = (:center,:center),color=:red,fontsize=20)
scatter!(ax2,8.06117,47.5542,color=:transparent,strokecolor=:blue,strokewidth=2,markersize=30)
text!(ax2,8.06117,50,text="(3,1,1)",align = (:center,:center),color=:blue,fontsize=20)
save("Figs/Fit_TF/Yaw_Dropout.png",f,px_per_unit=4)
f
##
Hi_fx = zeros(ComplexF64,18,length(moths))
Hf_fx = zeros(ComplexF64,18,length(moths))
Hi_yaw = zeros(ComplexF64,18,length(moths))
Hf_yaw = zeros(ComplexF64,18,length(moths))
Hi_roll = zeros(ComplexF64,18,length(moths))
Hf_roll = zeros(ComplexF64,18,length(moths))

for (i,m) in enumerate(moths)
    dat = allmoths[m]["ftpre"]
    datf = allmoths[m]["ftpos"]

    fpre = (allmoths[m]["stimpre"] .- mean(allmoths[m]["stimpre"])) .* pixel_conversion
    fpost = (allmoths[m]["stimpost"] .- mean(allmoths[m]["stimpost"])) .* pixel_conversion

    Hi_fx[:,i] = tf_freq(fpre,dat[:,1],freqqs,fs)
    Hf_fx[:,i] = tf_freq(fpost,datf[:,1],freqqs,fs)

    Hi_yaw[:,i] = tf_freq(fpre,dat[:,6],freqqs,fs)
    Hf_yaw[:,i] = tf_freq(fpost,datf[:,6],freqqs,fs)
    Hi_roll[:,i] = tf_freq(fpre,dat[:,5],freqqs,fs)
    Hf_roll[:,i] = tf_freq(fpost,datf[:,5],freqqs,fs)
end
lowyaw,hiyaw = get_all_tf_stats(Hi_yaw,Hf_yaw,freqqs; freq_max=8)
z = lowyaw.mg .* exp.(im .* lowyaw.mp)
zhi = hiyaw.mg .* exp.(im .* hiyaw.mp)

fun,ps,errors_low = any_given_model(3,2,1, z,freqqs[1:15];return_error=true)
funhigh,pshigh,errors_high = any_given_model(3,2,1,zhi,freqqs[1:15];return_error=true)

y_pred_all = zeros(100,15) |> x -> ComplexF64.(x)
y_pred_high_all = zeros(100,15) |> x -> ComplexF64.(x)
@showprogress for i in 1:100
    fun,ps = any_given_model(3,2,1, z,freqqs[1:15])
    funhigh,pshigh = any_given_model(3,2,1,zhi,freqqs[1:15])
    y_pred_all[i,:] = [fun(ps,f) for f in freqqs[1:15]]
    y_pred_high_all[i,:] = [funhigh(pshigh,f) for f in freqqs[1:15]]
end
##
y_pred = mean(y_pred_all, dims=1) |> x -> vec(x)
y_pred_high = mean(y_pred_high_all, dims=1) |> x -> vec(x)

mg_pred = abs.(y_pred)
mp_pred = angle.(y_pred)
f = Figure(size=(800,800))
ax = Axis(f[1,1],
    title = "Low Mass Yaw TF Model",
    xlabel="Frequency (Hz)",
    ylabel="Gain",
    yscale=log10,
    ylabelsize=20, yticklabelsize=15,
    xticklabelsize=15,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),
    limits=(0.1,10,nothing,nothing)
    )
# lines!(ax,freqqs[1:15],abs.(y_pred),u(go_neg(angle.color=:steelblue,linewidth=2,label="Low Model Prediction")
for i in 1:100
    lines!(ax,freqqs[1:15],abs.(y_pred_high_all[i,:]),color=:firebrick,linewidth=0.5,alpha=0.1)
    lines!(ax,freqqs[1:15],abs.(y_pred_all[i,:]),color=:steelblue,linewidth=0.5,alpha=0.1)
end
# lines!(ax,freqqs[1:15],abs.(y_pred_high),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax,freqqs[1:15],abs.(z),color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax,freqqs[1:15],abs.(z),abs.(z).-lowyaw.glow,lowyaw.ghigh.-abs.(z),color=:steelblue,)
scatter!(ax,freqqs[1:15],abs.(zhi),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax,freqqs[1:15],abs.(zhi),abs.(zhi).-hiyaw.glow,hiyaw.ghigh.-abs.(zhi),color=:firebrick,)
# Legend(f[1:2,2],ax)
ax2 = Axis(f[2,1],
    xlabel="Frequency (Hz)",
    ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=15,
    xticklabelsize=15,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),
    limits=(0.1,10,nothing,nothing)

    )
for i in 1:100
    lines!(ax2,freqqs[1:15],unwrap_negative(go_neg(angle.(y_pred_high_all[i,:]))),color=:firebrick,linewidth=0.5,alpha=0.1)
    lines!(ax2,freqqs[1:15],unwrap_negative(go_neg(angle.(y_pred_all[i,:]))),color=:steelblue,linewidth=0.5,alpha=0.1)
end
# lines!(ax2,freqqs[1:15],(unwrap_negative(go_neg(angle.(y_pred)))),color=:steelblue,linewidth=2,label="Low Model Prediction")
# lines!(ax2,freqqs[1:15],(unwrap_negative(go_neg(angle.(y_pred_high)))),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax2,freqqs[1:15],(unwrap_negative(go_neg(lowyaw.mp))), color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax2,freqqs[1:15],(unwrap_negative(go_neg(lowyaw.mp))),lowyaw.stp,color=:steelblue,)
scatter!(ax2,freqqs[1:15],(unwrap_negative(go_neg(hiyaw.mp))),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax2,freqqs[1:15],(unwrap_negative(go_neg(hiyaw.mp))),hiyaw.stp,color=:firebrick,)
# save("Figs/Fit_TF/Yaw_321_Tf.png",f,px_per_unit=4)
f
## 
""" Lets DO This for FX """
##
nps = Int.(range(1,5,5))
nzs = Int.(range(1,5,5))
ds  = [0,1]
sub = []
low_fx = DataFrame(np=Int[],nz=Int[],d=Int[],mean_error=Float64[],svmax=Float64[]) 
bar = Progress(length(nps)*length(nzs)*length(ds),desc="Fitting Models")
for np in nps 
    for nz in nzs
        for d in ds 
            if nz>np
                next!(bar)
            else
                mean_e,svmax = dropout_all(Hi_fx[1:15,:],np,nz,d)
                push!(low_fx,(np=np,nz=nz,d=d,mean_error=mean_e,svmax=svmax))
                next!(bar)
            end
        end
    end
end

high_fx = DataFrame(np=Int[],nz=Int[],d=Int[],mean_error=Float64[],svmax=Float64[]) 
bar = Progress(length(nps)*length(nzs)*length(ds),desc="Fitting Models")
sub = []
for np in nps 
    for nz in nzs
        for d in ds 
            if nz>np
                next!(bar)
            else
                mean_e,svmax = dropout_all(Hf_fx[1:15,:],np,nz,d)
                push!(high_fx,(np=np,nz=nz,d=d,mean_error=mean_e,svmax=svmax))
                next!(bar)
            end
        end
    end
end
##

f = Figure(size=(800,800)) 
ax = Axis(f[1,1],
    xlabel = "Max Singular Value of Parameters (a.u.)",
    ylabel = "Mean Dropout Error (a.u.)",
    title = "Low Mass Fx TF Model ",
    xscale = log10,    limits=(10^0,10^8,nothing,nothing))
scatter!(ax,low_fx.svmax,low_fx.mean_error,color=low_fx.d)
scatter!(ax,5.08959,42.2748,color=:transparent,strokecolor=:red,strokewidth=2,markersize=30)
scatter!(ax,10.418,50.9612,color=:transparent,strokecolor=:blue,strokewidth=2,markersize=30)
text!(ax,5.08959,45,text="(2,2,1)",align = (:center,:center),color=:red,fontsize=20)
text!(ax,10.418,53,text="(5,5,1)",align = (:center,:center),color=:blue,fontsize=20)
ax2 = Axis(f[2,1],
    xlabel = "Max Singular Value of Parameters (a.u.)",
    ylabel = "Mean Dropout Error (a.u.)",
    title = "High Mass Fx TF Model ",
    xscale = log10, limits=(10^0,10^8,nothing,nothing))
    # limits=(nothing,nothing,25,600))
scatter!(ax2,high_fx.svmax,high_fx.mean_error,color=high_fx.d)
scatter!(ax2,2.14929e7,56.4737,color=:transparent,strokecolor=:red,strokewidth=2,markersize=30)
text!(ax2,2.14929e7,59,text="(2,2,1)",align = (:center,:center),color=:red,fontsize=20)
scatter!(ax2,4.10884,48.2531,color=:transparent,strokecolor=:blue,strokewidth=2,markersize=30)
text!(ax2,4.10884,51,text="(5,5,1)",align = (:center,:center),color=:blue,fontsize=20)
save("Figs/Fit_TF/Fx_Dropout.png",f,px_per_unit=4)
f
##
lowfx,hifx = get_all_tf_stats(Hi_fx,Hf_fx,freqqs; freq_max=8)
z = lowfx.mg .* exp.(im .* lowfx.mp)
zhi = hifx.mg .* exp.(im .* hifx.mp)

fun,ps,errors_low = any_given_model(2,2,1, z,freqqs[1:15];return_error=true)
funhigh,pshigh,errors_high = any_given_model(2,2,1,zhi,freqqs[1:15];return_error=true)

y_pred_all = zeros(100,15) |> x -> ComplexF64.(x)
y_pred_high_all = zeros(100,15) |> x -> ComplexF64.(x)
@showprogress for i in 1:100
    fun,ps = any_given_model(5,5,1, z,freqqs[1:15])
    funhigh,pshigh = any_given_model(5,5,1,zhi,freqqs[1:15])
    y_pred_all[i,:] = [fun(ps,f) for f in freqqs[1:15]]
    y_pred_high_all[i,:] = [funhigh(pshigh,f) for f in freqqs[1:15]]
end
##
y_pred = [fun(ps,f) for f in freqqs[1:15]]
y_pred_high = [funhigh(pshigh,f) for f in freqqs[1:15]]
mg_pred = abs.(y_pred)
mp_pred = angle.(y_pred)
f = Figure(size=(800,800))
ax = Axis(f[1,1],
    title = "Fx TF Model",
    xlabel="Frequency (Hz)",
    ylabel="Gain",
    yscale=log10,
    ylabelsize=20, yticklabelsize=15,
    xticklabelsize=15,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),
    limits=(0.1,10,nothing,nothing)
    )
for i in 1:100
    lines!(ax,freqqs[1:15],abs.(y_pred_high_all[i,:]),color=:firebrick,linewidth=0.5,alpha=0.1)
    lines!(ax,freqqs[1:15],abs.(y_pred_all[i,:]),color=:steelblue,linewidth=0.5,alpha=0.1)
end
# lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue,linewidth=2,label="Low Model Prediction")
# lines!(ax,freqqs[1:15],abs.(y_pred_high),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax,freqqs[1:15],abs.(z),color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax,freqqs[1:15],abs.(z),abs.(z).-lowfx.glow,lowfx.ghigh.-abs.(z),color=:steelblue,)
scatter!(ax,freqqs[1:15],abs.(zhi),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax,freqqs[1:15],abs.(zhi),abs.(zhi).-hifx.glow,hifx.ghigh.-abs.(zhi),color=:firebrick,)
# Legend(f[1:2,2],ax)
ax2 = Axis(f[2,1],
    xlabel="Frequency (Hz)",
    ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=15,
    xticklabelsize=15,
    xticks=([0.1,1,10],[L"0.1",L"1",L"10"]),
    limits=(0.1,10,nothing,nothing)

    )
for i in 1:100
    lines!(ax2,freqqs[1:15],unwrap(go_neg(angle.(y_pred_high_all[i,:]))),color=:firebrick,linewidth=0.5,alpha=0.1)
    lines!(ax2,freqqs[1:15],unwrap(go_neg(angle.(y_pred_all[i,:]))),color=:steelblue,linewidth=0.5,alpha=0.1)
end
# lines!(ax2,freqqs[1:15],(unwrap(go_neg(angle.(y_pred)))),color=:steelblue,linewidth=2,label="Low Model Prediction")
# lines!(ax2,freqqs[1:15],(unwrap(go_neg(angle.(y_pred_high)))),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax2,freqqs[1:15],(unwrap(go_neg(lowfx.mp))), color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax2,freqqs[1:15],(unwrap(go_neg(lowfx.mp))),lowfx.stp,color=:steelblue,)
scatter!(ax2,freqqs[1:15],(unwrap(go_neg(hifx.mp))),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax2,freqqs[1:15],(unwrap(go_neg(hifx.mp))),hifx.stp,color=:firebrick,)
save("Figs/Fit_TF/FX_551_Tf.png",f,px_per_unit=4)
f