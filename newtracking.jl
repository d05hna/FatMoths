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
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
##

function process_data(file, fs, ftnames)
    df = h5_to_df(file)
    true_ft = DataFrame(transform_FT(transpose(Matrix(df[!, ftnames]))), ftnames)
    df[!, ftnames] = true_ft
    idx = findall(df.camtrig .> 2)[1]
    fx = df.fx[idx:idx+fs*20-1]
    return fx
end

function process_stim(stim_file)
    stim = CSV.read(stim_file, DataFrame)
    stim_20 = convert(Vector{Float64}, stim.filtered[1:6000])
    fftstim = fft(stim_20)[2:end]
    freq_range = fftfreq(length(stim_20), 300)[2:end]
    return freq_range, fftstim
end

function calculate_fft(data, fs)
    fft_data = fft(data)[2:end]
    fr_data = fftfreq(length(data), fs)[2:end]
    return fr_data, fft_data
end

function find_exact_indices(target_freqs, all_freqs)
    rounded_all_freqs = round.(all_freqs, digits=3)
    return [findfirst(x -> x == f, rounded_all_freqs) for f in target_freqs]
end

function calculate_gain(fx, stim)
    return log10.(abs.(fx) ./ abs.(stim))
end

function calculate_phase_offset(fx, stim)
    return angle.(fx) .- angle.(stim)
end

function freq_response(datadir, moth)
    files = glob("$(moth)/*.h5", datadir)
    stims = glob("*/*DLT*/*", datadir)
    fs = 10000
    ftnames = ["fx", "fy", "fz", "tx", "ty", "tz"]
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]

    # Pre-processing
    fx_pre = process_data(files[1], fs, ftnames)
    freq_range_stim_pre, fftstim_pre = process_stim(stims[4])
    fr_fx_pre, fftfx_pre = calculate_fft(fx_pre, fs)

    # Post-processing
    fx_post = process_data(files[3], fs, ftnames)
    freq_range_stim_post, fftstim_post = process_stim(stims[end])
    fr_fx_post, fftfx_post = calculate_fft(fx_post, fs)

    # Find exact indices for freqqs in stim and fx frequency ranges
    stim_indices_pre = find_exact_indices(freqqs, freq_range_stim_pre)
    stim_indices_post = find_exact_indices(freqqs, freq_range_stim_post)
    fx_indices_pre = find_exact_indices(freqqs, fr_fx_pre)
    fx_indices_post = find_exact_indices(freqqs, fr_fx_post)

    # Extract FFT values at frequencies of interest
    fft_stim_pre_freqqs = fftstim_pre[stim_indices_pre]
    fft_stim_post_freqqs = fftstim_post[stim_indices_post]
    fft_fx_pre_freqqs = fftfx_pre[fx_indices_pre]
    fft_fx_post_freqqs = fftfx_post[fx_indices_post]

    # Calculate gains and phase offsets
    gain_pre = calculate_gain(fft_fx_pre_freqqs, fft_stim_pre_freqqs)
    gain_post = calculate_gain(fft_fx_post_freqqs, fft_stim_post_freqqs)
    phase_pre = calculate_phase_offset(fft_fx_pre_freqqs, fft_stim_pre_freqqs)
    phase_post = calculate_phase_offset(fft_fx_post_freqqs, fft_stim_post_freqqs)

    # Create DataFrame with reshaped data
    result_df = DataFrame(
        freq = repeat(freqqs, 2),
        trial = repeat(["pre", "post"], inner=length(freqqs)),
        mag_fx = vcat(abs.(fft_fx_pre_freqqs), abs.(fft_fx_post_freqqs)),
        mag_stim = vcat(abs.(fft_stim_pre_freqqs), abs.(fft_stim_post_freqqs)),
        gain = vcat(gain_pre, gain_post),
        phase = rad2deg.(vcat(phase_pre, phase_post))
    )

    return result_df
end

##
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")
moths = ["2024_08_01","2024_08_12","2024_08_14","2024_08_15","2024_08_16"]
##
all_moths = DataFrame()
for moth in moths
    d = freq_response(datadir,moth)
    d.moth .= moth
    all_moths = vcat(all_moths,d,cols=:union)
end
save(joinpath(datadir,"mungedfreqdata.csv"),all_moths)
##


function quartiles(y)
    q1 = quantile(y, 0.25)
    median = quantile(y, 0.5)
    q3 = quantile(y, 0.75)
    return (q1 = q1, median = median, q3 = q3)
end
##
# Group the data by frequency and trial
grouped_data = combine(groupby(all_moths, [:freq, :trial]), :gain => quartiles => AsTable)
##
fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Log10 (Newtons / Position)",
    title = "Moth Frequency Response",
    xscale = log10
)

# Define colors for the two trials
colors = [:coral, :turquoise]
labels = ["Before Feeding","After Feeding"]

# Get unique trials
trials = unique(grouped_data.trial)

for (i, trial) in enumerate(trials)
    trial_data = filter(:trial => ==(trial), grouped_data)
    
    # Sort data by frequency to ensure correct line plotting
    sort!(trial_data, :freq)
    
    # Plot the median line
    lines!(ax, trial_data.freq, trial_data.median, 
           color=colors[i], label=labels[i], linewidth=2)
    
    # Plot the interquartile range
    band!(ax, trial_data.freq, trial_data.q1, trial_data.q3, 
          color=(colors[i], 0.3))
end

# Add legend
axislegend(ax, "Trial", position=:lt)

# Display the plot
display(fig)

save("mungedfigs/GAINS.png",fig,px_per_unit=4)

##

phase = combine(groupby(all_moths, [:freq, :trial]), :phase => quartiles => AsTable)
##

fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Phase Offset (def)",
    title = "Moth Frequency Response",
    xscale = log10
)

# Define colors for the two trials
colors = [:coral, :turquoise]
labels = ["Before Feeding","After Feeding"]

# Get unique trials
trials = unique(phase.trial)

for (i, trial) in enumerate(trials)
    trial_data = filter(:trial => ==(trial), phase)
    
    # Sort data by frequency to ensure correct line plotting
    sort!(trial_data, :freq)
    
    # Plot the median line
    lines!(ax, trial_data.freq, trial_data.median, 
           color=colors[i], label=labels[i], linewidth=2)
    
    # Plot the interquartile range
    band!(ax, trial_data.freq, trial_data.q1, trial_data.q3, 
          color=(colors[i], 0.3))
end

# Add legend
axislegend(ax, "Trial", position=:lt)

# Display the plot
display(fig)

save("mungedfigs/PHASE.png",fig,px_per_unit=4)