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
using Colors
using GLM
using SavitzkyGolay
using MultivariateStats
# using CairoMakie
using HypothesisTests
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:steelblue,:firebrick],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##
@load "fat_moths_set_1.jld" allmoths
##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

function get_yaw_tf(allmoths,moth)
    m = allmoths[moth]
    ypre = m["ftpre"][:,6]
    ypost = m["ftpos"][:,6]


    
    stimpre = m["stimpre"]

    stimpost = m["stimpost"]


    itp1 = interpolate(stimpre, BSpline(Linear()))
    stimpre_l = itp1(LinRange(1, length(itp1), Int(1e5)))

    itp2 = interpolate(stimpost, BSpline(Linear()))
    stimpost_l = itp2(LinRange(1, length(itp2), Int(1e5)))
    
    
    stimfreq = round.(fftfreq(100000,10000),digits=4)
    stimidx= [i for i in 1:length(stimpre) if stimfreq[i] in freqqs]


    coh_pre_pos = mt_coherence(hcat(ypre,stimpre_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]
    coh_post_pos = mt_coherence(hcat(ypost,stimpost_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]

    tfpre = (fft(ypre) ./ fft(stimpre_l))[stimidx]
    tfpost = (fft(ypost) ./ fft(stimpost_l))[stimidx]   

    return tfpre, tfpost,coh_pre_pos,coh_post_pos
end
##
function create_sampled_delay_matrix(signal::Vector{T}, num_rows::Int, step::Int) where T
    signal_length = length(signal)
    sampled_rows = collect(num_rows:-step:1)
    num_sampled = length(sampled_rows)
    num_cols = signal_length - (num_rows)
    result = Matrix{T}(undef, num_sampled, num_cols)
    
    @views for (i, r) in enumerate(sampled_rows)
        result[i, :] = signal[r:r+num_cols-1]
    end
    
    return result
end


##
moth = allmoths["2024_11_08"]
m = moth["data"]

m = add_relfx_column!(m)
m =  add_relyaw_column!(m)
put_stim_in!(m,allmoths,"2024_11_08")
m.wbtime = convert(Vector{Float64},m.wbtime)
##
ftpre = moth["ftpre"]
ftpost = moth["ftpos"]

itp1 = interpolate(moth["stimpre"], BSpline(Linear()))
itp2 = interpolate(moth["stimpost"], BSpline(Linear()))

stimpre = itp1(LinRange(1, length(itp1), Int(1e5))) # Interpolated stimulus for pre
stimpost = itp2(LinRange(1, length(itp2), Int(1e5))) # Interpolated stimulus for post

prewbtimes = unique(m[m.trial.=="pre",:wbtime])
prewbtimes = Int.(round.(prewbtimes .* 1e4))
postwbtimes = unique(m[m.trial.=="post",:wbtime])
postwbtimes = Int.(round.(postwbtimes .* 1e4))
##
ft_cca_pre = ftpre[Int(5e3)+1:end,:]'
for i in 1:6 
    ft_cca_pre[i,:] = zscore(ft_cca_pre[i,:])
end
ft_cca_post = ftpost[Int(5e3)+1:end,:]'

hist_stim_pre = create_sampled_delay_matrix(stimpre,Int(5e3),100)

##
ccPre = fit(MultivariateStats.CCA,ft_cca_pre,hist_stim_pre,outdim=1)



lat = convert(Vector{Float64},predict(ccPre,ftpre',:x)[:])

ccPost = fit(MultivariateStats.CCA,ftpost',stimpost')
lat2 = convert(Vector{Float64},predict(ccPost,ftpost',:x)[:])
##
mo = "2024_11_08"
moths =  ["2024_11_11","2024_11_20","2024_11_07","2024_11_08","2025_01_30","2024_11_01","2024_11_05","2024_11_04"]
ftnames = ["fx","ty","tz"]
cohs = DataFrame()
for mo in moths 
    for ft in ftnames
        coh_pre_pos,coh_post_pos,_,_ = transfer_function_coherence(allmoths,mo;axis=ft)
        tmp = Dict(
            "axis" => ft,
            "moth" => mo,
            "pre" => mean(coh_pre_pos[1]),
            "post" => mean(coh_post_pos[1])

        )
        push!(cohs,tmp,cols=:union)
    end
end
st_cohs = stack(cohs,Not(:axis,:moth),variable_name="trial",value_name="coh")
##