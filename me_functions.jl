function h5_to_df(path)
    h5open(path, "r") do file
        # Read the matrix from the file
        matrix = read(file, "data")
        
        # Read the column names from the file
        column_names = vec(read(file, "names"))
        
        # Create a DataFrame using the matrix and column names
        df = DataFrame(matrix, column_names,makeunique=true)

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