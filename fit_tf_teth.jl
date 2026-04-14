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
    dat = allmoths[m]["ftpre"] .* 1000
    datf = allmoths[m]["ftpos"] .* 1000
    
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

fun,ps,errors_low = any_given_model(2,2,1, z,freqqs[1:15];return_error=true)
funhigh,pshigh,errors_high = any_given_model(2,2,1,zhi,freqqs[1:15];return_error=true)
"""
ps =   0.35490947873108636
  0.1152084914700482
 -3.8684026017022557
  0.3979934438135263
  0.02145305713830685
  0.26227141175732266

pshigh =   0.0027828503177715954
  0.0006147739527836687
 -1.5835970926745957
  0.11912904583810387
  1.6502817017754015
0.8652894089992633
"""


##
# y_pred = mean(y_pred_all, dims=1) |> x -> vec(x)
# y_pred_high = mean(y_pred_high_all, dims=1) |> x -> vec(x)
ps =   [0.35490947873108636,
  0.1152084914700482,
 -3.8684026017022557,
  0.3979934438135263,
  0.02145305713830685 * 1000,
  0.26227141175732266,]

pshigh =   [0.0027828503177715954,
  0.0006147739527836687,
 -1.5835970926745957,
  0.11912904583810387,
  1.6502817017754015 * 1000,
  .8652894089992633,]
y_pred = [fun(ps,f) for f in freqqs[1:15]]
y_pred_high = [funhigh(pshigh,f) for f in freqqs[1:15]]
mg_pred = abs.(y_pred)
mp_pred = angle.(y_pred)
f = Figure(size=(800,800))
ax = Axis(f[1,1],
    title = "Low Mass Yaw TF Model",
    # xlabel="Frequency (Hz)",
    ylabel="Gain (mNmm / mm )",
    yscale=log10,
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,0.8,20),
    yticks = ([1,10],[L"1",L"10"])
    )
lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue,linewidth=2,label="Low Model Prediction")

lines!(ax,freqqs[1:15],abs.(y_pred_high),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax,freqqs[1:15],abs.(z),color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax,freqqs[1:15],abs.(z),abs.(z).-lowyaw.glow,lowyaw.ghigh.-abs.(z),color=:steelblue,)
scatter!(ax,freqqs[1:15],abs.(zhi),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax,freqqs[1:15],abs.(zhi),abs.(zhi).-hiyaw.glow,hiyaw.ghigh.-abs.(zhi),color=:firebrick,)
ax2 = Axis(f[2,1],
    xlabel="Frequency (Hz)",
    ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,nothing,nothing),
    yticks=([0,-2pi,-4pi],["0",L"-2\pi",L"-4\pi"])

    )

lines!(ax2,freqqs[1:15],unwrap(((angle.(y_pred)))),color=:steelblue,linewidth=2,label="Low Model Prediction")
lines!(ax2,freqqs[1:15],unwrap_negative(((angle.(y_pred_high)))),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax2,freqqs[1:15],unwrap((angle.(z))), color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax2,freqqs[1:15],unwrap((angle.(z))),lowyaw.stp,color=:steelblue,)
scatter!(ax2,freqqs[1:15],unwrap_negative((angle.(zhi))),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax2,freqqs[1:15],unwrap_negative((angle.(zhi))),hiyaw.stp,color=:firebrick,)
save("Figs/Fit_TF/Yaw_221_Tf.png",f,px_per_unit=4)
f
## 
""" Lets DO This for FX """
##
lowfx,hifx = get_all_tf_stats(Hi_fx,Hf_fx,freqqs; freq_max=8)
z = lowfx.mg .* exp.(im .* lowfx.mp)
zhi = hifx.mg .* exp.(im .* hifx.mp)

fun,ps,errors_low = any_given_model(2,2,1, z,freqqs[1:15];return_error=true)
funhigh,pshigh,errors_high = any_given_model(2,2,1,zhi,freqqs[1:15];return_error=true)


##
# Good P Values 
ps = [
        0.0025855361903374926,
        0.000781992722175423,
        -7.472119380033888,
        0.8867365245987195,
        -0.19461666433931227 * 1000,
        0.27272843068979924,
    ]

pshigh = [ 
        0.0025855361903374926,
        0.0005581992722175423,
        -8.172119380033888,
        0.5867365245987195,
        -0.28461666433931227 * 1000,
        0.6217727143068979924,
    ]

y_pred = [fun(ps,f) for f in freqqs[1:15]]
y_pred_high = [funhigh(pshigh,f) for f in freqqs[1:15]]
mg_pred = abs.(y_pred)
mp_pred = angle.(y_pred)
f = Figure(size=(800,800))
ax = Axis(f[1,1],
    title = "Fx TF Model",
    # xlabel="Frequency (Hz)",
    ylabel="Gain (mN / mmm )",
    yscale=log10,
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,10^-1.4,1),
    yticks = ([0.1,1],[L"0.1",L"1"]),
    )

lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue,linewidth=2,label="Low Model Prediction")
lines!(ax,freqqs[1:15],abs.(y_pred_high),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax,freqqs[1:15],abs.(z),color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax,freqqs[1:15],abs.(z),abs.(z).-lowfx.glow,lowfx.ghigh.-abs.(z),color=:steelblue,)
scatter!(ax,freqqs[1:15],abs.(zhi),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax,freqqs[1:15],abs.(zhi),abs.(zhi).-hifx.glow,hifx.ghigh.-abs.(zhi),color=:firebrick,)
# Legend(f[1:2,2],ax)
ax2 = Axis(f[2,1],
    xlabel="Frequency (Hz)",
    ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,nothing,nothing),
    yticks = ([0,-2pi,-4pi],["0",L"-2\pi",L"-4\pi"])
    )

lines!(ax2,freqqs[1:15],((unwrap(angle.(y_pred)))),color=:steelblue,linewidth=2,label="Low Model Prediction")
lines!(ax2,freqqs[1:15],((unwrap_negative(angle.(y_pred_high) ))),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax2,freqqs[1:15],(unwrap(angle.(z))), color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax2,freqqs[1:15],(unwrap(angle.(z))),lowfx.stp,color=:steelblue,)
scatter!(ax2,freqqs[1:15] ,unwrap_negative(angle.(zhi)),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax2,freqqs[1:15],(unwrap_negative(angle.(zhi))),hifx.stp,color=:firebrick,)
save("Figs/Fit_TF/FX_551_Tf.png",f,px_per_unit=4)
f
##
## 
""" Lets DO This for ROLL """
##
lowroll,hiroll = get_all_tf_stats(Hi_roll,Hf_roll,freqqs; freq_max=8)
z = lowroll.mg .* exp.(im .* lowroll.mp)
zhi = hiroll.mg .* exp.(im .* hiroll.mp)

fun,ps,errors_low = any_given_model(2,2,1, z,freqqs[1:15];return_error=true)
funhigh,pshigh,errors_high = any_given_model(2,2,1,zhi,freqqs[1:15];return_error=true)


##
# Good P Values 
ps = [
        1.0874811208249495
        0.1638031372988557
        0.04075293309940653
        -0.0033523504448844943
        0.20213489295015155
        0.26274900373295584
    ]

pshigh = [ 
        1.0874811208249495
        0.1638031372988557
        0.04075293309940653
        -0.0033523504448844943
        0.25213489295015155
        0.51274900373295584
    ]

y_pred = [fun(ps,f) for f in freqqs[1:15]]
y_pred_high = [funhigh(pshigh,f) for f in freqqs[1:15]]
mg_pred = abs.(y_pred)
mp_pred = angle.(y_pred)
f = Figure(size=(800,800))
ax = Axis(f[1,1],
    title = "Roll TF Model",
    # xlabel="Frequency (Hz)",
    ylabel="Gain (mNmm / mmm )",
    yscale=log10,
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,10^-0.1,10^1.35),
    yticks = ([1,10],[L"1",L"10"]),
    )

lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue,linewidth=2,label="Low Model Prediction")
lines!(ax,freqqs[1:15],abs.(y_pred_high),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax,freqqs[1:15],abs.(z),color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax,freqqs[1:15],abs.(z),abs.(z).-lowroll.glow,lowroll.ghigh.-abs.(z),color=:steelblue,)
scatter!(ax,freqqs[1:15],abs.(zhi),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax,freqqs[1:15],abs.(zhi),abs.(zhi).-hiroll.glow,hiroll.ghigh.-abs.(zhi),color=:firebrick,)
# Legend(f[1:2,2],ax)
ax2 = Axis(f[2,1],
    xlabel="Frequency (Hz)",
    ylabel="Phase (radians)",
    ylabelsize=20, yticklabelsize=22,
    xticklabelsize=22,
    xticks=([0.1,1,8],[L"0.1",L"1",L"8"]),
    limits=(0.1,9,nothing,nothing),
    yticks = ([0,-2pi,-4pi],["0",L"-2\pi",L"-4\pi"])
    )

lines!(ax2,freqqs[1:15],((unwrap(angle.(y_pred)))),color=:steelblue,linewidth=2,label="Low Model Prediction")
lines!(ax2,freqqs[1:15],((unwrap_negative(angle.(y_pred_high) ))),color=:firebrick,linewidth=2,label="High Model Prediction")
scatter!(ax2,freqqs[1:15],(unwrap_negative(angle.(z))), color=:steelblue,markersize=8,label="Low Data")
errorbars!(ax2,freqqs[1:15],(unwrap_negative(angle.(z))),lowroll.stp,color=:steelblue,)
scatter!(ax2,freqqs[1:15] ,unwrap_negative(angle.(zhi)),color=:firebrick,markersize=8,label="High Data")
errorbars!(ax2,freqqs[1:15],(unwrap_negative(angle.(zhi))),hiroll.stp,color=:firebrick,)
save("Figs/Fit_TF/Roll_Final.png",f,px_per_unit=4)
f