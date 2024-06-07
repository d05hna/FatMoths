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
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
##
datadir = "/home/doshna/Documents/PHD/data/fatties/"

files = glob("*_06/*.h5",datadir)
ps = glob("*_06/*.csv",datadir)
dpath = files[1]
spath = ps[1]
##

stim = CSV.read(spath,DataFrame)
##
function h5_to_df(path)
    h5open(path, "r") do file
        # Read the matrix from the file
        matrix = read(file, "data")
        
        # Read the column names from the file
        column_names = vec(read(file, "names"))
        
        # Create a DataFrame using the matrix and column names
        df = DataFrame(matrix, column_names)

        mapss = Dict(
            "ch1" => "fx",
            "ch2" => "fy",
            "ch3" => "fz",
            "ch4" => "tx",
            "ch5" => "ty",
            "ch6" => "tz" )
        if "ch1" in names(df)
            rename!(df,mapss)
        end

            
        return df
        end
    end
##
df = h5_to_df(dpath)

##
en = stim.Time[end]
beg = stim.Time[end]-60 

thestim = stim[stim.Time.>beg,:]
thestim.Time = thestim.Time .- thestim.Time[1]
##
ftnames = ["fx","fy","fz","tx","ty","tz"]
##
itp = interpolate(thestim.Position,BSpline(Linear()))
            
full_pos = itp(LinRange(1, length(itp), size(df)[1]))
##
true_ft = DataFrame(transform_FT(transpose(Matrix(df[!,ftnames]))),ftnames)
##

function remove_bias(moth,datadir)
    params = Dict{String, Any}()

    q = h5_to_df(glob("$(moth)/*quietpost.h5",datadir)[1])
    empty = h5_to_df(glob("$(moth)/*empty.h5",datadir)[1])
    ft_names = ["fx","fy","fz","tx","ty","tz"]

    bias = mean(Matrix(empty[!,ft_names]), dims=1)
    quiet = mean(Matrix(q[!,ft_names]), dims=1)

    quiet = transform_FT(transpose(quiet .- bias))

    A = [0 -quiet[3] quiet[2];
    quiet[3] 0 -quiet[1];
    -quiet[2] quiet[1] 0]
    B = -quiet[4:6]

    func(x) = norm(A*x-B)
    sol = optimize(func, [0.0,0.0,0.0])
    COM = Optim.minimizer(sol) # (x, y, z coordinates of most likely center of mass)
    M_transform = [1 0 0 0 0 0;
                0 1 0 0 0 0;
                0 0 1 0 0 0;
                0 COM[3] -COM[2] 1 0 0;
                -COM[3] 0 COM[1] 0 1 0;
                COM[2] -COM[1] 0 0 0 1]
    params["mass"] = quiet[3] / 9.81 * 1000 # N to g
    params["COM"] = COM
    params["M_transform"] = M_transform
    params["bias"] = bias
    return(params)
end
##
# Create filters
fs = 10000

z_bandpass = [5, 30]
ft_lowpass = 1000

cheby_bandpass = digitalfilter(Bandpass(z_bandpass...; fs=fs), Chebyshev1(4, 4))
butter_lowpass = digitalfilter(Lowpass(ft_lowpass; fs=fs), Butterworth(4))

select!(df,ftnames)
df.time = - range(nrow(df)/fs, step=- 1/fs, length=nrow(df))
df.moth .= split(dpath,"/")[end][1:10]
df.trial = map(x -> x < -40 ? "pre" : (x < -20 ? "mid" : "post"), df.time)
df.wb = @pipe filtfilt(cheby_bandpass, df.fz) |>
            hilbert .|>
            angle .|> 
            sign |>
            (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
            cumsum(_ .> 0.1)
        local maxhold = last(df.wb) # Hold value for shifting wingbeat numbers
##
spikes_mat = get_amps_sort(joinpath(datadir,"2024_06_06"))
sorted_trials = unique(spikes_mat[:,1])

##
muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
for (mi,m) in enumerate(muscle_names)
    local column = zeros(Bool, nrow(df))
    local inds =(spikes_mat[:,2] .== mi - 1) .&&
        (spikes_mat[:,6] .== 1)
    # Shift from -maxtime:0 to 0:+maxtime
    spikes_mat[inds,3] .+= abs(df.time[1])
    column[round.(Int, spikes_mat[inds, 3] * fs)] .= 1
    df[:, m] = column
end

##

df.time = round.(df.time, digits=4)
##



##
df = @pipe df |> 
    # Column for wingbeat length
    groupby(_, :wb) |> 
    transform(_, :wb => (x -> length(x) / fs) => :wblen) |> 
        #     # Mark wingbeats to remove 
        #     # (using bool column instead of removing now to allow proper spike unwrapping)
        #     # Remove wingbeats if too short or too long 
           groupby(_, :wb) |> 
             @transform(_, :validwb = (first(:wblen) >= 1/30) .& (first(:wblen) <= 1/10)) |>
            # Remove wingbeats where power muscles don't fire
             groupby(_, :wb) |> 
                 @transform(_, :validwb = ifelse.(
                any(:ldlm) .&& any(:rdlm) .&& :validwb, 
             true, false)) 
        #     # Go ahead and actually remove wingbeats of length 1 because that breaks a lot
        #     groupby(_, :wb) |> 
        #     combine(_, x -> nrow(x) == 1 ? DataFrame() : x)
        # # If there's no valid wingbeats in this trial, skip the heavier computations and move on
        if sum(df.validwb) == 0
            print("oops")
        end
##
df = @pipe df |> 
    # Make phase column
    groupby(_, :wb) |>
    transform(_, :wb => (x -> LinRange(0, 1, length(x))) => :phase) |> 
    # Make time from start of wb column
    rename!(_, :time => :time_abs) |> 
    groupby(_, :wb) |> 
    transform(_, :time_abs => (x -> LinRange(0, length(x)/fs, length(x))) => :time)
##

"""
Stimulus response
"""
##
pre_pos = full_pos[1:200000]
pre_tz = df.tz[1:200000]

post_pos = full_pos[400001:end]
post_tz = df.tz[400001:end]

freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70,16.70 ,19.90]

##

ft_pre_pos = fft(pre_pos)
ft_pre_tz = fft(pre_tz)

N = length(pre_pos)

gainphasepre = DataFrame()
tmp = Dict()

for f in freqqs
    index = round(Int, f * N / fs) + 1
    gain = abs(ft_pre_tz[index]) / abs(ft_pre_pos[index])
    gain_dB = 20 * log10(gain)

    phase_shift = angle(ft_pre_tz[index]) - angle(ft_pre_pos[index])
    phase_shift_degrees = phase_shift * 180 / π

    tmp["freq"] = f 
    tmp["gain"] = gain_dB
    tmp["phase"] = phase_shift
    push!(gainphasepre, tmp,cols=:union)
end
##
ft_post_pos = fft(post_pos)
ft_post_tz = fft(post_tz)
gainphasepost = DataFrame()
for f in freqqs
    index = round(Int, f * N / fs) + 1
    gain = abs(ft_post_tz[index]) / abs(ft_post_pos[index])
    gain_dB = 20 * log10(gain)

    phase_shift = angle(ft_post_tz[index]) - angle(ft_post_pos[index])
    phase_shift_degrees = phase_shift * 180 / π

    tmp["freq"] = f 
    tmp["gain"] = gain_dB
    tmp["phase"] = phase_shift
    push!(gainphasepost, tmp,cols=:union)
end

## Bode Plot

freq_range = 0.1:1:500 # Hz

magnitude_dB = zeros(length(freq_range))
phase_degrees = zeros(length(freq_range))

t = LinRange(0,20,length(pre_pos))

for (i, freq) in enumerate(freq_range)
    # Calculate the Fourier coefficients at the current frequency
    X = sum(pre_pos .* exp.(-im * 2π * freq * t))
    Y = sum(pre_tz .* exp.(-im * 2π * freq * t))
    
    # Calculate magnitude and phase
    magnitude_dB[i] = 20 * log10(abs(Y) / abs(X))
    phase_degrees[i] = angle(Y) - angle(X)
end

dB_post = zeros(length(freq_range))
phase_post = zeros(length(freq_range))

t = LinRange(0,20,length(pre_pos))

for (i, freq) in enumerate(freq_range)
    # Calculate the Fourier coefficients at the current frequency
    X = sum(post_pos .* exp.(-im * 2π * freq * t))
    Y = sum(post_tz .* exp.(-im * 2π * freq * t))
    
    # Calculate magnitude and phase
    dB_post[i] = 20 * log10(abs(Y) / abs(X))
    phase_post[i] = angle(Y) - angle(X)
end
##