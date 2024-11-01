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

function read_ind(datadir,moth,wb_method)
        ftnames = ["fx","fy","fz","tx","ty","tz"]
        muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
        
        files = readdir(joinpath(datadir,moth))
    
        lb = [-10,0,50]
        ub = [10,0,60]
    
        spikes_mat = get_amps_sort(joinpath(datadir,moth))
        sorted_trials = unique(spikes_mat[:,1])
        ## Get the params
        params=Dict{Any,Any}()
        mt = filter(s -> occursin("empty", s), files)[1]
    
        quiets = filter(s -> occursin("quiet",s),files)
        qpre = quiets[1]
        qpost = quiets[end]
        empty = h5_to_df(joinpath(datadir,moth,mt))
        bias = mean(Matrix(empty[!,ftnames]),dims=1)
        quiet = mean(Matrix(h5_to_df(joinpath(datadir,moth,qpre))[!,ftnames]),dims=1)
        quietpost = mean(Matrix(h5_to_df(joinpath(datadir,moth,qpost))[!,ftnames]),dims=1)
        ## THE COM STUFF IS NOT WORKING :(
        quiet = transform_FT(transpose(quiet .- bias))
        quietpost = transform_FT(transpose(quietpost .- bias))
    
        A = [0 -quiet[3] quiet[2];
                quiet[3] 0 -quiet[1];
                -quiet[2] quiet[1] 0]
            B = -quiet[4:6]
            func(x) = norm(A*x-B)
            sol = optimize(func, lb, ub,  lb .+ (ub .- lb) ./ 2)
            COM = Optim.minimizer(sol) # (x, y, z coordinates of most likely center of mass)
            M_transform = [1 0 0 0 0 0;
                        0 1 0 0 0 0;
                        0 0 1 0 0 0;
                        0 COM[3] -COM[2] 1 0 0;
                        -COM[3] 0 COM[1] 0 1 0;
                        COM[2] -COM[1] 0 0 0 1]
            # Put main outputs of moth paramters into single dict
        params["FT_names"] = ftnames
        params["EMG_names"] = muscle_names
        params["fs"] = round(Int, 10000)
        params["mass"] = quiet[3] / 9.81 * 1000 # N to g
        params["COM"] = COM
        params["M_transform"] = M_transform
        params["bias"] = bias
        params["mass_post"] = quietpost[3] / 9.81 * 1000
    
        ##
    
        maxwb = 0 
        full_data = DataFrame()
    
        for t in sorted_trials
            dpath = joinpath(datadir,moth,moth*"_00$(Int(t)).h5")
            local df = h5_to_df(dpath)
            df[!,ftnames] = transform_FT(transpose(Matrix(df[!,ftnames]) .- params["bias"]),params["M_transform"])
            
            fs = 10000
            
            z_bandpass = [5, 30]
            ft_lowpass = 1000
            
            cheby_bandpass = digitalfilter(Bandpass(z_bandpass...; fs=fs), Chebyshev1(4, 4))
            
            ##
            
            select!(df,"time",ftnames)
            df.time = round.(df.time .- df.time[end] .- 1/fs, digits=4)
            df.moth .= moth
            df.trial .= t 
            df.species .= "Manduca sexta"
            
            ##
            for (mi,m) in enumerate(muscle_names)
                local column = zeros(Bool, nrow(df))
                global inds = (spikes_mat[:,1] .== df.trial[1]) .&& 
                    (spikes_mat[:,2] .== mi - 1) .&&
                    (spikes_mat[:,6] .== 1)
                # Shift from -maxtime:0 to 0:+maxtime
                spikes_mat[inds,3] = spikes_mat[inds,3].+ abs(df.time[1])
                indss = round.(Int,spikes_mat[inds,3]*params["fs"])
                tmp = filter(x -> x <= nrow(df), indss)
                column[tmp] .= 1
                df[:, m] = column
            end
            ## wb by methods
            if wb_method == "rdlm"
                df.wb = cumsum(df.rdlm)
            elseif wb_method == "ldlm"
                df.wb = cumsum(df.ldlm)
            elseif wb_method == "hilbert"
                df.wb = @pipe filtfilt(cheby_bandpass, df.fz) |>
                hilbert .|>
                angle .|> 
                sign |>
                (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
                cumsum(_ .> 0.1)
            end
        
            
            df.time = round.(df.time,digits=4)
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
                        @transform(_, :validwb = (first(:wblen) >= 1/30) .& (first(:wblen) <= 1/15)) |>
                        # Remove wingbeats where power muscles don't fire
                        groupby(_, :wb) |> 
                            @transform(_, :validwb = ifelse.(
                            any(:ldlm) .&& any(:rdlm) .&& :validwb, 
                        true, false))|>
                groupby(_, :wb) |> 
                        combine(_, x -> nrow(x) == 1 ? DataFrame() : x)
                    # # If there's no valid wingbeats in this trial, skip the heavier computations and move on
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
            df.wb = df.wb .+ maxwb
    
    
            full_data = vcat(full_data,df)
            maxwb = maximum(full_data.wb)
        end
        return(full_data,params)
    end
##

function get_side_slips(datadir,moth,params,trials)
    ftnames = ["fx","fy","fz","tx","ty","tz"]
    files = glob("*.h5",joinpath(datadir,moth))
    tris =[path for path in files if occursin("00", path)]
    path_pre = tris[trials[1]]
    path_post = tris[trials[2]]

    pre_fx = transform_FT(transpose(Matrix(h5_to_df(path_pre)[!,ftnames]) .- params["bias"]))[:,1]
    post_fx = transform_FT(transpose(Matrix(h5_to_df(path_post)[!,ftnames]) .- params["bias"]))[:,1]

    return(pre_fx,post_fx)
end
##
