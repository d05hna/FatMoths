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
using ProgressMeter
using Statistics 
using StatsBase
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/functions.jl")
include("/home/doshna/Documents/PHD/comparativeMPanalysis_bmd/readAndPreprocessFunctions.jl")
include("me_functions.jl")

##
@load "fat_moths_set_1.jld" allmoths
##
moths = collect(keys(allmoths))
##
function get_mat_for_vahid(allmoths,m)
    muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
    df = allmoths[m]["data"]
    pr = df[df.trial.=="pre",:]
    po = df[df.trial.=="post",:]

    pr.time_abs .-= floor(minimum(pr.time_abs))
    po.time_abs .-= floor(minimum(po.time_abs))

    smpre = zeros(Int(1e5),12)
    smpo = zeros(Int(1e5),12)
    ##

    for (i, mn) in enumerate(muscle_names)
        idpr = round.(Int,pr[pr.muscle.==mn,:time_abs] .* 1e4)
        idpo = round.(Int,po[po.muscle.==mn,:time_abs] .* 1e4)

        smpre[idpr,i] .= 1 
        smpo[idpo,i] .= 1 
    end

    ##
    itp1 = interpolate(allmoths[m]["stimpre"], BSpline(Linear()))
    itp2 = interpolate(allmoths[m]["stimpost"], BSpline(Linear()))
    itp3 = interpolate(allmoths[m]["velpre"], BSpline(Linear()))
    itp4 = interpolate(allmoths[m]["velpost"], BSpline(Linear()))

    smpre[:,11] = itp1(LinRange(1, length(itp1), Int(1e5)))
    smpre[:,12] = itp3(LinRange(1, length(itp1), Int(1e5)))

    smpo[:,11] = itp2(LinRange(1, length(itp1), Int(1e5)))
    smpo[:,12] = itp4(LinRange(1, length(itp1), Int(1e5)))

    smpre = hcat(smpre,allmoths[m]["ftpre"])
    smpo = hcat(smpo,allmoths[m]["ftpos"])
    ##
    z_bandpass = [5, 30]
    ft_lowpass = 1000

    cheby_bandpass = digitalfilter(Bandpass(z_bandpass...), Chebyshev1(4, 4);fs=1e4)
    dfpre = DataFrame(smpre,vcat(muscle_names,["stim","stimvel","fx","fy","fz","tx","ty","tz"]))
    dfpre.wb = @pipe filtfilt(cheby_bandpass, dfpre.fz) |>
        hilbert .|>
        angle .|> 
        sign |>
        (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
        cumsum(_ .> 0.1)

    dfpo = DataFrame(smpo,vcat(muscle_names,["stim","stimvel","fx","fy","fz","tx","ty","tz"]))
    dfpo.wb = @pipe filtfilt(cheby_bandpass, dfpo.fz) |>
        hilbert .|>
        angle .|> 
        sign |>
        (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
        cumsum(_ .> 0.1)

    return dfpre,dfpo
end

##
mc = get_mean_changes(allmoths;axis=6)
moths = mc[mc.mean_gain .> 25,:moth]
##
dir = "/home/doshna/Desktop/Vahid_MP/"

for m in moths 
    pr,po = get_mat_for_vahid(allmoths,m)
    CSV.write(joinpath(dir,"$(m)_LOW.csv"),pr)
    CSV.write(joinpath(dir,"$(m)_HIGH.csv"),po)
end

