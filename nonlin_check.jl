using MAT
using CSV
using JLD2 
using FFTW
using GLMakie 
using DataFrames
using ColorSchemes
using Pipe
using DataFramesMeta
using StatsBase
using Interpolations
using DSP
using MultivariateStats
GLMakie.activate!()
theme = theme_minimal()
theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)
include("me_functions.jl")
@load "fat_moths_set_1.jld" allmoths


freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]
fs = Int(1e4)
##

mc = get_mean_changes(allmoths;axis=6)
moths = ["2024_11_01","2024_11_08","2024_11_11","2025_03_20","2025_04_02","2025_09_19","2025_10_10"]
n_moths = length(moths)
Ci = zeros(18,n_moths) |> x -> Complex.(x)
Cf = zeros(18,n_moths) |> x -> Complex.(x)
for (i,m) in enumerate(moths)
    println(m)
    skinny = allmoths[m]["ftpre"]
    fat = allmoths[m]["ftpos"]

    stimpre = allmoths[m]["stimpre"]
    itp = interpolate(stimpre, BSpline(Linear()))
    sprel = itp(LinRange(1, length(itp), Int(1e5)))

    stimpost = allmoths[m]["stimpost"]
    itp = interpolate(stimpost, BSpline(Linear()))
    spostl = itp(LinRange(1,length(itp),Int(1e5)))

    sfx = zscore(skinny[:,1])
    syaw = zscore(skinny[:,6])
    sroll = zscore(skinny[:,5])

    ffx = zscore(fat[:,1])
    fyaw = zscore(fat[:,6])
    froll = zscore(fat[:,5])

    s = hcat(sfx,syaw,sroll)

    mo = fit(MultivariateStats.PCA,s')
    println("Low VE: ", mo.prinvars[1] / mo.tvar)

    ci = (s * mo.proj)
     
    f = hcat(ffx,fyaw,froll)
    mo = fit(MultivariateStats.PCA,f')
    # println("High VE: ", mo.prinvars[1] / mo.tvar)

    cf = (f * mo.proj)

    # ci = sfx .+ (-1 .* syaw)  #.+ sroll 
    # cf = ffx .+ (-1 .* fyaw)  #.+ froll
 
    stimpre = zscore(allmoths[m]["stimpre"])

    stimpost = zscore(allmoths[m]["stimpost"])

    Ci[:,i] = tf_freq(stimpre,ci,freqqs,fs)
    Cf[:,i] = tf_freq(stimpost,cf,freqqs,fs)
end
##
i = 1 
for i in 1:length(freqqs)
    F = Figure(size=(600,600))
    ax = Axis(F[1,1],xlabel="real",ylabel = "imaginary",title="$(freqqs[i]) Hz",
    limits=(-1.5,1.5,-1.5,1.5))
    scatter!(ax,real.(Ci[i,:]),imag.(Ci[i,:]),color=:steelblue)
    scatter!(ax,real.(Cf[i,:]),imag.(Ci[i,:]),color=:firebrick)
    save("Figs/NONLINCHECK/complex_$(freqqs[i]).png",F,px_per_unit=4)
end
##
for i in 1:n_moths 
    F = Figure(size=(800,800))
    ax1 = Axis(F[1,1],ylabel = "gain",title=moths[i])
    lines!(ax1, freqqs,abs.(Ci[:,i]),color=:steelblue,linewidth=3)
    lines!(ax1,freqqs,abs.(Cf[:,i]),color=:firebrick,linewidth=3)
    hidexdecorations!(ax1)
    F
    ax2 = Axis(F[2,1],ylabel="phase",xlabel="Frequency")
    linkxaxes!(ax2,ax1) 
    lines!(ax2,freqqs,unwrap(angle.(Ci[:,i])),linewidth=3,color=:steelblue) 
    lines!(ax2,freqqs,unwrap(angle.(Cf[:,i])),linewidth=3,color=:firebrick)
    ax2.yticks = ([0,-pi,-2pi,-3pi],[L"0",L"-\pi",L"-2\pi",L"-3\pi"])
    save("Figs/NONLINCHECK/Bode/$(moths[i]).png",F,px_per_unit=4)
end
##
mci = mean(Ci,dims=2)[:]
mcf = mean(Cf,dims=2)[:]

f = Figure() 
ax = Axis(f[1,1],xscale=log10,yscale=log10,
ylabel="Gain",yticks=[0.01,0.1,1],title="Mean Tethered Tracking Response")
hidexdecorations!(ax)
lines!(ax,freqqs,abs.(mci),color=:steelblue,linewidth=3)
lines!(ax,freqqs,abs.(mcf),color=:firebrick,linewidth=3)
ax2 = Axis(f[2,1],xscale=log10,
ylabel="Phase",xlabel="Frequencey",xticks=[0.2,1,10],yticks=([0,-pi,-2pi,-3pi],[L"0",L"-\pi",L"-2\pi",L"-3\pi"]))
lines!(ax2,freqqs,unwrap(angle.(mci)),color=:steelblue,linewidth=3)
lines!(ax2,freqqs,unwrap(angle.(mcf)),color=:firebrick,linewidth=3)
linkxaxes!(ax2,ax)
f