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
##
datadir = "/home/doshna/Documents/PHD/data/fatties/"

files = glob("*_20/*.h5",datadir)
ps = glob("*_20/*.csv",datadir)
dpath = files[1]
# spath = ps[1]
##

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


function read_moth_emgft(dpath,moth)
    df = h5_to_df(dpath)
    fx = df.fx
    ##
    ftnames = ["fx","fy","fz","tx","ty","tz"]

    ##
    true_ft = DataFrame(transform_FT(transpose(Matrix(df[!,ftnames]))),ftnames)
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
    df.species .= "Manduca sexta"
    df.wb = @pipe filtfilt(cheby_bandpass, df.fz) |>
                hilbert .|>
                angle .|> 
                sign |>
                (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
                cumsum(_ .> 0.1)
            local maxhold = last(df.wb) # Hold value for shifting wingbeat numbers
    ##
    spikes_mat = get_amps_sort(joinpath(datadir,moth))
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
    phase_wrap_thresh =Dict(
        "Manduca sexta" => Dict("ax"=>2.0, "ba"=>0.5, "sa"=>0.9, "dvm"=>0.4, "dlm"=>0.8),
    )   


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
        transform(_, :time_abs => (x -> LinRange(0, length(x)/fs, length(x))) => :time)  |>
        groupby(_, :wb) |> 
            transform(_, Symbol.(ftnames) .=> mean .=> Symbol.(ftnames)) |> 
            # Keep only where spikes happened to save on memory
            @subset(_, :lax .|| :lba .|| :lsa .|| :ldvm .|| :ldlm .|| :rdlm .|| :rdvm .|| :rsa .|| :rba .|| :rax) |> 
            # Pivot spike time columns to longer form
            stack(_, Symbol.(muscle_names)) |>
            rename!(_, :variable => :muscle, :value => :spike) |> 
            @subset(_, :spike .== 1) |> 
            # Remove columns no longer useful now with just spikes
            select!(_, Not(:spike)) |> 
            # Perform shifting of spikes, remove any that couldn't be matched to a wingbeat
            transform!(_, [:moth, :species, :muscle, :wb, :wblen, :time, :phase, :validwb] => unwrap_spikes_to_next => [:wb, :wblen, :time, :phase]) |> 
            # Remove invalid wingbeats
            @subset(_, :validwb) |> 
            select!(_, Not(:validwb)) |> 
            @subset(_, (!).(isnan.(:time))) |> 
            # Column for wingbeat frequency
            groupby(_, :wb) |> 
            @transform(_, :wbfreq = 1 ./ :wblen)
    ##
    return(df,fx)
end
##
"""
Stimulus response
"""
##
moth = "2024_06_20"
df,fx = read_moth_emgft(dpath,moth)
##
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70,16.70 ,19.90]

##
params = Dict{String, Any}()

q = h5_to_df(glob("$(moth)/*quiet.h5",datadir)[1])
empty = h5_to_df(glob("$(moth)/*empty.h5",datadir)[1])
ft_names = ["fx","fy","fz","tx","ty","tz"]

post = h5_to_df(glob("$(moth)/*quietpost.h5",datadir)[1])

bias = mean(Matrix(empty[!,ft_names]), dims=1)
quiet = mean(Matrix(q[!,ft_names]), dims=1)

quiet = transform_FT(transpose(quiet .- bias))

post = mean(Matrix(post[!,ft_names]), dims=1)
post = transform_FT(transpose(post .- bias))

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
params["mass_post"] = post[3] / 9.81 * 1000
params["COM"] = COM
params["M_transform"] = M_transform
params["bias"] = bias




##
params = remove_bias(moth,datadir)

# mass_offset = params["mass"] - 1.5

# params["mass"] = params["mass"] - mass_offset
# params["mass_post"] = params["mass_post"] - mass_offset
##

stim = CSV.read(spath,DataFrame)
en = stim.Time[end]
beg = stim.Time[end]-60 

thestim = stim[stim.Time.>beg,:]
thestim.Time = thestim.Time .- thestim.Time[1]
##
itp = interpolate(thestim.Position,BSpline(Linear()))
            
full_pos = itp(LinRange(1, length(itp), size(df)[1]))
##

function calculate_gains(signal1, signal2, fs, frequencies)
    fft1 = fft(signal1)
    fft2 = fft(signal2)

    freq_vector = fftfreq(length(signal1), fs)

    gains = zeros(length(frequencies))

    for (i, frequency) in enumerate(frequencies)
        freq_index = argmin(abs.(freq_vector .- frequency))
        mag1 = abs(fft1[freq_index])
        mag2 = abs(fft2[freq_index])
        gains[i] = 20*log10(mag1 /mag2)
    end

    return gains
end
##
fs = 10000

fpre = df[df.trial .== "pre",[:time_abs,:fx,:fy,:fz]]
rename!(fpre,"time_abs"=>"time")
# ppre = zscore(full_pos[1:200000])


fpost = df[df.trial .== "post",[:time_abs,:fx,:fy,:fz]]
rename!(fpost,"time_abs"=>"time")
# ppost = zscore(full_pos[400001:end])

f_range = sort(vcat(freqqs, freqqs .-0.05,freqqs .+ 0.05))
##
gs_pre = calculate_gains(zscore(fpre.fx),ppre,fs,freqqs)
gs_post = calculate_gains(zscore(fpost.fx),ppost,fs,freqqs)

max_value = maximum([maximum(abs.(gs_pre)), maximum(abs.(gs_post))])

##

f = Figure(figsize=(6,6))
ax = Axis(f[1,1],xscale=log10)

pre = scatter!(ax,freqqs, gs_pre,alpha=0.5)
post = scatter!(ax,freqqs,gs_post,alpha=0.5)

ax.xlabel = "Freq"
ax.ylabel = "Gain (db)"

ax.title = "Frequency Response Before and after feeding"

fr = vlines!(ax,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

Legend(f[1,2],[pre,post,fr],["First Epoch","Last Epoch","Driving Freqs"])

hidedecorations!(ax,label=false,ticklabels=false,ticks=false)

save("gain$(moth).png",f,px_per_unit=4)
f
##

function integrate(vec,dt=0.0001)
    itrs = length(vec)

    res = zeros(itrs)

    for i in 2:itrs
        res[i] = res[i-1] + vec[i]*dt
    end
    return res
end
##

##
Ax = (fpre.fx .- mean(fpre.fx))./params["mass"]
Vx = integrate(Ax)
X = integrate(Vx)

sl = (X[end] - X[1])/length(X) 
fl = (1:1:length(X)) .* sl

detrend_X = (X - fl) 

##

t = fpre.time
f = Figure()
ax1 = Axis(f[1,1])
ax2 = Axis(f[2,1])
ax3 = Axis(f[3,1])

lines!(ax1,t,Ax)
lines!(ax2,t,Vx)
lines!(ax3,t,detrend_X)

ax1.title = "Acceleration (x)"
ax2.title = "Velocity (x)"
ax3.title = "(Detrend) Position (x)"
ax3.xlabel = "Time (s)"

save("kinematics.png",f,px_per_unit=4)

f
##
dt = 0.0001
# motor_pos = full_pos[400001:end]
# motor_velo = [(motor_pos[i]-motor_pos[i-1])/dt for i in 2:length(motor_pos)]
# pushfirst!(motor_velo,0)

# fft_motor = fft(motor_pos)
# fft_velo = fft(motor_velo)
# fft_fx = fft(fpost.fx)

freq_range = fftfreq(length(fx[1:200000]),fs)[2:10000]

# mag= abs.(fft_motor)[2:10000]
# phase = angle.(fft_motor)[2:10000]

# magvelo = abs.(fft_velo)[2:10000]
# phasevelo = angle.(fft_velo)[2:100x00]

magfx = abs.(fft(fx[400001:end]))[2:10000]
phasefx = angle.(fft(fx[400001:end]))[2:10000]

##

f = Figure()
ax1 = Axis(f[1,1],xscale=log10)
# ax2 = Axis(f[2,1],xscale=log10)

# ax3 = Axis(f[1,2],xscale=log10)
# ax4 = Axis(f[2,2],xscale=log10)

# motor = lines!(ax1,freq_range,mag ./maximum(mag),color=:blue,alpha=0.7)
fr = vlines!(ax1,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
# lines!(ax2,freq_range,phase,color=:blue,alpha=0.7)

forc = lines!(ax1,freq_range,magfx ./maximum(magfx),color=:green,alpha=0.7)
# scatter!(ax2,freq_range,phasefx,color=:green,alpha=0.7)
# lines!(ax3,freq_range,magvelo ./maximum(magvelo),color=:orange)
# fr = vlines!(ax3,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
# lines!(ax4,freq_range,phasevelo,color=:orange)

ax1.title= "Moth Magnitude After Feeding"
ax2.title= "Moth Phase"
# ax3.title= "Motor Veloctiy Magnitude"
# ax4.title= "Motor Velocity Phase"

Legend(f[:,2],[forc],["Moth Fx"])
save("FFTPost.png",f,px_per_unit=4)
f
## Plot the WbFrequency by Epoch 
d4t = unique(select(df,:wb,:wblen,:trial,:validwb))
d4t = d4t[d4t.validwb,Not(:validwb)]
d4t.freq = 1 ./ d4t.wblen

plot = data(d4t)*mapping(:freq,color=:trial)*histogram(bins=75)*visual(alpha=0.7)
f = draw(plot,axis=(; title="$moth Wb Freq by Epoch"))
# save("wbfreq.png",f,px_per_unit=4)
f
## Lets Look at phase by Epoch

mu = df[df.rdlm .|| df.ldlm,:]

select!(mu,:wb,:trial,:ldlm,:rdlm,:phase)

new = stack(mu, [:ldlm, :rdlm], variable_name="muscle", value_name="fires")
new = new[new.fires,Not(:fires)]
new = new[new.trial .!= "mid",:]

plot = data(new) * mapping(:phase,row=:muscle,color=:trial) * histogram(bins=75)*visual(alpha=0.7) 
f = draw(plot,figure = (; title="$moth DLM Phase by Epock"))
# save("DLMPhase.png",f,px_per_unit=4)
f
##

first_second = df.fz[1:10000]
fs_ldlm = df.rdlm[1:10000]
z_bandpass = [5, 30]
ft_lowpass = 1000

cheby_bandpass = digitalfilter(Bandpass(z_bandpass...; fs=10000), Chebyshev1(4, 4))
butter_lowpass = digitalfilter(Lowpass(ft_lowpass; fs=10000), Butterworth(4))

cheb = filtfilt(cheby_bandpass,first_second)
hil = hilbert(cheb)
ang = angle.(hil)
sig = sign.(ang)

count = pushfirst!(diff(sig),0)

# (x -> pushfirst!(diff(x), 0)) |>
#  cumsum(_ .> 0.1)

wb = @pipe filtfilt(cheby_bandpass,first_second) |>
    hilbert .|>
    angle .|>
    sign |>
    (x -> pushfirst!(diff(x),0)) |> 
    cumsum(_ .> 0.1)

plop = diff(wb)
change_index = findall(!iszero,plop) .+ 1
spike_idx = (findall(!iszero,diff(fs_ldlm)) .+ 1) ./10000

time_wb_change = change_index ./10000

time = LinRange(0,1,10000)
#
fig = Figure()
ax = Axis(fig[1,1])

lines!(ax,time,angle.(hilbert(filtfilt(cheby_bandpass,first_second))),color=:green)
# vlines!(ax,time_wb_change,color=:blue,alpha=0.3)
scatter!(ax,spike_idx,zeros(length(spike_idx)),color=:red,alpha=0.4)
fig