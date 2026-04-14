using DSP 
using FFTW 
using JLD2 
using GLMakie 
using DataFrames 
using AlgebraOfGraphics 
using StatsBase
using Pipe 
using DataFramesMeta  
using Interpolations
using Optim
using ControlSystems
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moths_set_1.jld" allmoths 
include("me_functions.jl")
mc = get_mean_changes(allmoths;axis=6)
moths = mc[mc.mean_gain .> 25,:moth]
fs = Int(1e4)
N = Int(1e5)
freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
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


##

m = moths[3] 
tz_pre = tf_freq(-0.14 .*(allmoths[m]["stimpre"] .- mean(allmoths[m]["stimpre"])),allmoths[m]["ftpre"][:,6] .* 1000,freqqs,fs)

np = 2
nz = 2

mf,ps = any_given_model(np,nz,1,tz_pre[1:15],freqqs)
ps[end]=round(ps[end],digits=3)
##
y_pred = [mf(ps,f) for f in freqqs[1:15]]
f = Figure() 
ax = Axis(f[1,1],xscale=log10,yscale=log10,xlabel="freq",ylabel="gain")
scatter!(ax,freqqs[1:15],abs.(tz_pre)[1:15],color=:steelblue)
lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue)
ax2 = Axis(f[2,1],xscale=log10,ylabel="phase",xlabel="freq")
scatter!(ax2,freqqs[1:15],unwrap(angle.(tz_pre)[1:15]),color=:steelblue)
lines!(ax2,freqqs[1:15],unwrap(angle.(y_pred)),color=:steelblue)

f
##
function build_statespace(mod_fun, ps, nz, np)
    a = ps[1:nz]           # numerator coeffs
    b = ps[nz+1:nz+np]  # denominator coeffs  
    println(a)
    println(b)
    k = ps[end-1]
    T = ps[end]
    
    num = k .* reverse(b)
    den = reverse(a)
    
    num = num ./ den[1]
    den = den ./ den[1]
    s = tf("s")
    L = abs(T)
    return tf(num, den) * exp(-L*s)
end

 

x = 0.14 .* (allmoths[m]["stimpre"] .- mean(allmoths[m]["stimpre"]))
itp = interpolate(x, BSpline(Linear()))
flower = itp(LinRange(1, length(itp), Int(1e5)))


tf_ss = build_statespace(nothing,ps,nz,np)
tf_ss_d = c2d(tf_ss,1/fs);

result = lsim(tf_ss_d,flower');
y,t,x,u=result;


##
y_pred = [mf(ps,f) for f in freqqs[1:15]]
f = Figure(size=(1200,600)) 

ax = Axis(f[1,1],xscale=log10,yscale=log10,xlabel="freq",ylabel="gain")
scatter!(ax,freqqs[1:15],abs.(tz_pre)[1:15],color=:steelblue)
lines!(ax,freqqs[1:15],abs.(y_pred),color=:steelblue)
ax2 = Axis(f[2,1],xscale=log10,ylabel="phase (deg)",xlabel="freq")
scatter!(ax2,freqqs[1:15],unwrap(angle.(tz_pre)[1:15]) .* 360/2pi,color=:steelblue)
lines!(ax2,freqqs[1:15],unwrap(angle.(y_pred)) .*360/2pi,color=:steelblue)
ax = Axis(f[1:2,2],xlabel="time (s)",ylabel="flower position (mm)")
lines!(ax,t,flower,color=:black)
ax2 = Axis(f[1:2,2], ylabel="Yaw Torque mN * mm ", yaxisposition=:right,yticklabelcolor=:steelblue,ylabelcolor=:steelblue)
hidexdecorations!(ax2)
lines!(ax2,t,vec(y),color=:steelblue)
f
