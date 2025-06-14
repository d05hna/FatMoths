function h5_to_dict(path)
    data = Dict{String,Any}()
    h5open(path,"r") do file
        for name in keys(file)
            data[name] = read(file[name])
        end
    end
    return data
end
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
##
function compute_PCA_time_ben(args...; nPC=3)
    ftnames = ["fx", "fy", "fz", "tx", "ty", "tz"]
    outdf = DataFrame(
        Matrix{Float64}(undef, length(args[1]), nPC*length(ftnames)),
        [f * "_pc" * string(s) for s in 1:3 for f in ftnames]
    )
    
    uniquewb = unique(args[7])
    wbi = group_indices(args[7])
    reps = [length(wbi[w]) for w in uniquewb]
    valid = [args[8][wbi[w][1]] for w in uniquewb]
    shortest_wblen = findmin(reps[valid])[1]
    
    for (i,ft) in enumerate(ftnames)
        mat = Matrix{Float64}(undef, sum(valid), shortest_wblen)
        
        for (j,w) in enumerate(uniquewb[valid])
            mat[j,:] = args[i][wbi[w]][1:shortest_wblen]
        end
        # println(mat)
        # Correct PCA fit and transform
        M = fit(PCA, transpose(mat))
        scores = Matrix{Float64}(undef, length(uniquewb), nPC)
        scores[valid,:] = transform(M, mat')[:,:nPC]'
        
        # Fill invalid wingbeats with zeros instead of undef
        scores[.!valid,:] .= 0


        for s in 1:nPC
            outdf[:,Symbol(ft * "_pc" * string(s))] = inverse_rle(scores[:,s], reps)
        end
    end
    return outdf
end

##

function read_ind(datadir,moth,wb_method)
        ftnames = ["fx","fy","fz","tx","ty","tz"]
        muscle_names = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
        
        files = readdir(joinpath(datadir,moth))
    
        lb = [-10,0,50]
        ub = [10,0,60]
    
        spikes_mat = Matrix(DataFrame(get_amps_sort(joinpath(datadir,moth))))
        # println(spikes_mat)
        sorted_trials = unique(spikes_mat[:,1])
        ## Get the params
        params=Dict{Any,Any}()
        mt = filter(s -> occursin("empty", s), files)[1]
    
        quiets = filter(s -> occursin("quiet",s),files)
        qpre = quiets[1]
        qpost = quiets[end]
        empty = h5_to_df(joinpath(datadir,moth,mt))[Int(end-1e5+1):end,:]
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
            dpath = joinpath(datadir,moth,moth[1:10]*"_00$(Int(t)).h5")
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
                DataFramesMeta.transform(_, :wb => (x -> length(x) / fs) => :wblen) |> 
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
                DataFramesMeta.transform(_, :wb => (x -> LinRange(0, 1, length(x))) => :phase) |> 
                # Make time from start of wb column
                rename!(_, :time => :time_abs) |> 
                groupby(_, :wb) |> 
                DataFramesMeta.transform(_, :time_abs => (x -> LinRange(0, length(x)/fs, length(x))) => :time)  |>
                groupby(_, :wb) |> 
                DataFramesMeta.transform(_, Symbol.(ftnames) .=> mean .=> Symbol.(ftnames)) |> 
                    # Keep only where spikes happened to save on memory
                    @subset(_, :lax .|| :lba .|| :lsa .|| :ldvm .|| :ldlm .|| :rdlm .|| :rdvm .|| :rsa .|| :rba .|| :rax) |> 
                    # Pivot spike time columns to longer form
                    stack(_, Symbol.(muscle_names)) |>
                    rename!(_, :variable => :muscle, :value => :spike) |> 
                    @subset(_, :spike .== 1) |> 
                    # Remove columns no longer useful now with just spikes
                    select!(_, Not(:spike)) |> 
                    # Perform shifting of spikes, remove any that couldn't be matched to a wingbeat
                    DataFramesMeta.transform!(_, [:moth, :species, :muscle, :wb, :wblen, :time, :phase, :validwb] => unwrap_spikes_to_next => [:wb, :wblen, :time, :phase]) |> 
                    DataFramesMeta.transform(_, 
                    Symbol.(vcat(ftnames, "wb", "validwb")) => 
                    compute_PCA_time => 
                        # compute_PCA_phase => 
                        Symbol.([f * "_pc" * string(s) for s in 1:3 for f in ftnames]))
            df = @pipe df |>
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
    path_pre = tris[Int(trials[1])]
    path_post = tris[Int(trials[2])]

    pre_fx = transform_FT(transpose(Matrix(h5_to_df(path_pre)[!,ftnames]) .- params["bias"]))[:,1:6]
    post_fx = transform_FT(transpose(Matrix(h5_to_df(path_post)[!,ftnames]) .- params["bias"]))[:,1:6]

    return(pre_fx,post_fx)
end
##
function get_tracking_fig(pre,post,fs=1e4)
    time = range(0,10,length(pre))

    fftpre = abs.(fft(pre)[2:50000]) 
    fftpost = abs.(fft(post)[2:50000])
    freqrange = fftfreq(length(pre),fs)[2:50000]
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

    id = findfirst(x->x ==15, freqrange)
    fftpre = fftpre[1:id] ./ length(pre)
    fftpost=fftpost[1:id] ./ length(post)
    freqrange=freqrange[1:id]


    f = Figure(resolution=(800,800))

    ax1 = Axis(f[1,1],xlabel="Time",ylabel="Side Slip Force",title="Forces")
    ax2 = Axis(f[1,2],xlabel="Time",ylabel="Side Slip Force",title="Forces")
    ax3 = Axis(f[2, 1], 
        xlabel = "Frequency", 
        ylabel = "Amplitude",
        title = "FFT",
        xscale = log10,
        limits=(nothing,12,nothing,nothing)
        )
    ax4 = Axis(f[2, 2], 
        xlabel = "Frequency", 
        ylabel = "Amplitude",
        title = "FFT",
        xscale = log10,
        limits=(nothing,12,nothing,nothing)
        )
    ax5 = Axis(f[3, :], 
        xlabel = "Frequency", 
        ylabel = "Amplitude",
        xscale = log10,
        limits=(nothing,12,nothing,nothing)
        )

    pre = lines!(ax1,time,pre,color=:turquoise)
    post = lines!(ax2,time,post,color=:coral)
    lines!(ax3,freqrange,fftpre,color=:turquoise)
    lines!(ax4,freqrange,fftpost,color=:coral)

    lines!(ax5,freqrange,fftpre,color=:turquoise)
    lines!(ax5,freqrange,fftpost,color=:coral)

    vlines!(ax3,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
    vlines!(ax4,freqqs,color=:grey,linestyle=:dash,alpha=0.3)
    vlines!(ax5,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

    Legend(f[:,3],[pre,post],["Before Feeding","After Feeding"])
    return(f)
end
##
function gain_by_freq_fig(changes)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Frequency", ylabel="Gain Change (%)",xscale=log10)

    mean_fz = [(moth=m, mean_fz=mean(filter(r -> r.moth == m, changes).fz_change)) 
            for m in unique(changes.moth)]
    mean_fz_values = [x.mean_fz for x in mean_fz]

    norm_positions = (mean_fz_values .- minimum(mean_fz_values)) ./ 
                (maximum(mean_fz_values) - minimum(mean_fz_values))

    for (i, (moth, mean_fz)) in enumerate(mean_fz)
    moth_data = filter(r -> r.moth == moth, changes)
    color = cgrad(:viridis)[norm_positions[i]]
    lines!(ax, moth_data.freq, moth_data.gain_change, 
            color=color, label=moth)
    scatter!(ax, moth_data.freq, moth_data.gain_change, 
                color=color)
    end

    Colorbar(fig[1, 2], limits=(minimum(mean_fz_values), maximum(mean_fz_values)),
            colormap=:viridis, label="Mean FZ Change (%)")
    axislegend()
    return(fig)
end
##
function muscle_count(group,muscle_names)
    counts = Dict(muscle_name => 0 for muscle_name in muscle_names)
    for row in eachrow(group)
        counts[row.muscle] +=1
    end
    return counts
end
##
function get_big_data(df,torp) 
    """
    Takes in a df with moth wb trial muscle and phase and outputs the spike counts and phases for every possible spike 
    The max number of spikes you see for a given muscle is the number of collumns that muscle will get for phase
    if sometimes the lax spikes 5 times but a wb only has 2 spikes 3-5 will be populated with missing
    """
    ## Make sure there are no repeated wb accross moths 
    sort!(df,:moth)
    maxwb = 0

    for moth in unique(df.moth)
        mask = df.moth .== moth
        df[mask,:wb] .+= maxwb
        maxwb = maximum(df[mask,:wb])   
    end
    muscle_names = unique(df.muscle) 
    ##

    ## Get the Muscle Count Data For all 10 

    grouped = groupby(df,[:moth,:wb,:trial])

    pivoted_data = []
    for g in grouped 
        counts = muscle_count(g,muscle_names)
        moth,wb,trial = g[1,:moth],g[1,:wb],g[1,:trial]
        row = [moth, wb, trial, [counts[muscle] for muscle in muscle_names]...]
        push!(pivoted_data, row)
    end
    ##
    count_df = DataFrame(permutedims(hcat(pivoted_data...)),[:moth,:wb,:trial,Symbol.(muscle_names .*"_count")...])
    ##Find the Maximum possible number of spikes per muscle 
    max_counts = Dict()
    for mus in muscle_names
        max_counts[mus] = maximum(count_df[!,mus*"_count"])
    end
    ## Get the phase data, missing 
    if torp == "phase"
        pivoted_data = []
        for g in grouped 
            moth,wb,trial = g[1,:moth],g[1,:wb],g[1,:trial]
            phases = Dict(mus => [] for mus in muscle_names)

            for row in eachrow(g)
                mus = row.muscle
                phase = row.phase
                push!(phases[mus],phase)
            end

            for mus in muscle_names
                phases[mus] = sort(phases[mus])
                while length(phases[mus]) < max_counts[mus]
                    push!(phases[mus],missing)
                end
            end
            row = [moth,wb,trial]
            for mus in muscle_names 
                for i in 1:max_counts[mus]
                    push!(row, phases[mus][i])
                end
            end
            push!(pivoted_data,row)
        end

        cols = [:moth,:wb,:trial]
        for mus in muscle_names 
            for i in 1:max_counts[mus]
                push!(cols,Symbol("$(mus)$(i)"))
            end
        end

        phasedf = DataFrame(permutedims(hcat(pivoted_data...)),cols)
    end
    ##
    if torp == "time"
        pivoted_data = []
        for g in grouped 
            moth,wb,trial = g[1,:moth],g[1,:wb],g[1,:trial]
            times = Dict(mus => [] for mus in muscle_names)

            for row in eachrow(g)
                mus = row.muscle
                time = row.time
                push!(times[mus],time)
            end

            for mus in muscle_names
                times[mus] = sort(times[mus])
                while length(times[mus]) < max_counts[mus]
                    push!(times[mus],missing)
                end
            end
            row = [moth,wb,trial]
            for mus in muscle_names 
                for i in 1:max_counts[mus]
                    push!(row, times[mus][i])
                end
            end
            push!(pivoted_data,row)
        end

        cols = [:moth,:wb,:trial]
        for mus in muscle_names 
            for i in 1:max_counts[mus]
                push!(cols,Symbol("$(mus)$(i)"))
            end
        end

        phasedf = DataFrame(permutedims(hcat(pivoted_data...)),cols)
    end
    ##

    full_data = leftjoin(count_df,phasedf,on=[:wb,:trial,:moth])
    fzs = unique(select(df, [:wb, :trial, :moth, :fz,:wbfreq]), [:wb, :trial, :moth])
    leftjoin!(full_data,fzs,on=[:wb,:trial,:moth])
    select!(full_data,:moth,:wb,:trial,:fz,:wbfreq,:)
    return(full_data)
end
##
function get_mean_changes(allmoths;axis=1)
    fs = 1e4
    moths = collect(keys(allmoths))
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
    peaks = DataFrame()
    all_data = DataFrame()
    for moth in moths 

        d = allmoths[moth]["data"]
        notpc = filter(col -> !contains(string(col), "_pc"), names(d))
        sel = filter(x -> !(x in ["wbtime","pos","vel"]), notpc)
        all_data=vcat(all_data,d[!,sel])
    end
    for moth in moths 
        pre = allmoths[moth]["ftpre"][:,axis]
        post = allmoths[moth]["ftpos"][:,axis]

        fftpre = abs.(fft(pre)[2:50000])
        fftpost = abs.(fft(post)[2:50000])
        freqrange = round.(fftfreq(Int(1e5),fs)[2:50000],digits=4)
        println(freqrange[1:20])
        d4t = all_data[all_data.moth.==moth,:]
        for f in freqqs
            id = findfirst(x -> x == f, freqrange)

            peakpre = fftpre[id]
            peakpost = fftpost[id]
            prdic = Dict("moth"=>moth,"freq" => f, "trial" => "pre", 
                "peak" => peakpre, "fz" => mean(d4t[d4t.trial.=="pre",:fz]) )
            podic = Dict("moth"=>moth,"freq" => f, "trial" => "post", 
                "peak" => peakpost,"fz" => mean(d4t[d4t.trial.=="post",:fz]) )
            push!(peaks,prdic,cols=:union)
            push!(peaks,podic,cols=:union)
        end
    end
    peaks.fz = peaks.fz .* -1 
    ##
    ##
    g=9.81
    ms = Dict(
        "2024_11_01" => Dict("pre"=>1.912,"post"=>2.075),
        "2024_11_04" => Dict("pre"=>2.149,"post"=>2.289),
        "2024_11_05" => Dict("pre"=>2.190,"post"=>2.592),
        "2024_11_07" => Dict("pre"=>1.801,"post"=>1.882),
        "2024_11_08" => Dict("pre"=>2.047,"post"=>2.369),
        "2024_11_11" => Dict("pre"=>1.810,"post"=>2.090),
        "2024_11_20" => Dict("pre"=>1.512,"post"=>1.784),
        "2025_01_30" => Dict("pre"=>2.13, "post"=>2.546),
        "2025_03_20" => Dict("pre"=>2.246, "post"=>2.344),
        "2025_01_14" => Dict("pre" => 2.07, "post"=> 2.531),
        "2025_04_02" => Dict("pre" => 1.7, "post"=> 1.916)
      )
    ##
    function normalize_fz(row, ms, g)
        return row.fz / ((ms[row.moth]["pre"]/1000) * g)
    end

    peaks.norm_fz = map(row -> normalize_fz(row, ms, g), eachrow(peaks))

    ##
    grouped = groupby(peaks,[:moth,:freq])

    changes = combine(grouped) do gdf 
        pre_vals = filter(r -> r.trial == "pre",gdf)
        post_vals = filter(r -> r.trial == "post",gdf)
        
        (
            fz_change =  (mean(post_vals.norm_fz) - mean(pre_vals.norm_fz)) / abs(mean(pre_vals.norm_fz)),
            gain_change = (mean(post_vals.peak) - mean(pre_vals.peak))/abs(mean(pre_vals.peak))*100
        )
    end

    mean_changes = combine(groupby(changes, :moth),
    :fz_change => mean => :mean_fz,
    :gain_change => mean => :mean_gain
    )
    mean_changes.mass .= 0.
    mean_changes.mass_change.=0.
    for row in eachrow(mean_changes)
        row.mass_change = ms[row.moth]["post"] - ms[row.moth]["pre"] 
        row.mass = ms[row.moth]["pre"]
    end
    ##
    return(mean_changes)
end
function put_stim_in!(df,allmoths,moth)
    spre = allmoths[moth]["stimpre"]
    spost = allmoths[moth]["stimpost"]
    stimtime = range(0,10,length=length(spre))

    sgpre = savitzky_golay(spre,11,4;deriv=0)
    vpre = savitzky_golay(spre, 11, 4; deriv=1)

    sgpost = savitzky_golay(spost,11,4;deriv=0)
    vpost = savitzky_golay(spost,11,4;deriv=1)

    dfpre = df[df.trial.=="pre",:]
    dfpre.time_abs .-= minimum(dfpre.time_abs)
    dfprewbtime = combine(groupby(dfpre,:wb),
        :time_abs => mean => :wbtime
    )
    prestimidx = [findlast(x-> x  <= i,stimtime) for i in dfprewbtime.wbtime]

    dfprewbtime.pos = sgpre.y[prestimidx]
    dfprewbtime.vel = vpre.y[prestimidx]
    dfprewbtime.wb = unique(dfpre.wb)

    dfpost = df[df.trial.=="post",:]
    dfpost.time_abs .-= minimum(dfpost.time_abs)
    dfpostwbtime = combine(groupby(dfpost,:wb),
        :time_abs => mean => :wbtime
    )
    poststimidx = [findlast(x-> x  <= i,stimtime) for i in dfpostwbtime.wbtime]
    dfpostwbtime.pos = sgpost.y[poststimidx]
    dfpostwbtime.vel = vpost.y[poststimidx]
    dfpostwbtime.wb = unique(dfpost.wb)

    allstim = vcat(dfprewbtime,dfpostwbtime)

    leftjoin!(df,allstim,on=:wb)
    select!(df, Not(r"^[ft][xyz]_pc[4-9]|[ft][xyz]_pc10$"))
end

function transfer_function(allmoths, moth;axis="fx")
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
    ft_ref = Dict(
        "fx" => 1,
        "fy" => 2,
        "fz" => 3,
        "tx" => 4,
        "ty" => 5,
        "tz" => 6
    )
    
    stimpre = allmoths[moth]["stimpre"]
    fxpre = allmoths[moth]["ftpre"][:,ft_ref[axis]]

    stimpost = allmoths[moth]["stimpost"]
    fxpost = allmoths[moth]["ftpos"][:,ft_ref[axis]]


    fftprestim = (fft(float.(stimpre)))
    stimfreq = round.(fftfreq(length(stimpre),300),digits=1)
    fftpoststim = (fft(float.(stimpost)))

    fftfxpre = fft(float.(fxpre))
    fftfxpos = fft(float.(fxpost)) 
    fxfreq = round.(fftfreq(length(fxpre),1e4),digits=1)

    frset = Set(freqqs)

    stimidx= [i for i in 1:length(stimpre) if stimfreq[i] in frset]
    fxidx= [i for i in 1:length(fxpre) if fxfreq[i] in frset]

    sfxpre = fftfxpre[fxidx]
    sfxpost = fftfxpos[fxidx]

    sstimpre = fftprestim[stimidx]
    sstimpost = fftpoststim[stimidx]

    pre = sfxpre ./ sstimpre
    post = sfxpost ./ sstimpost
    return pre,post
end
function add_relpos_column(df)
    groups = groupby(df, [:moth, :trial])
    
    result = DataFrame()
    
    for group in groups
        sorted_pos = sort(group.pos)
        n = length(sorted_pos)
        
        lower_bound = sorted_pos[ceil(Int, n/3)]
        upper_bound = sorted_pos[ceil(Int, 2*n/3)]
        
        group_copy = copy(group)
        
        group_copy.relpos = map(group_copy.pos) do val
            if val <= lower_bound
                return "right"
            elseif val > upper_bound
                return "left"
            else
                return "middle"
            end
        end
        append!(result, group_copy)
    end
    
    return result
end
function add_relfx_column!(df)
    groups = groupby(df, [:moth, :trial])
    
    result = DataFrame()
    
    for group in groups
        sorted_pos = sort(group.fx)
        n = length(sorted_pos)
        
        lower_bound = sorted_pos[ceil(Int, n/3)]
        upper_bound = sorted_pos[ceil(Int, 2*n/3)]
        
        group_copy = copy(group)
        
        group_copy.relfx = map(group_copy.fx) do val
            if val <= lower_bound
                return "right"
            elseif val > upper_bound
                return "left"
            else
                return "middle"
            end
        end
        append!(result, group_copy)
    end
    
    return result
end
function add_relyaw_column!(df)
    groups = groupby(df, [:moth, :trial])
    
    result = DataFrame()
    
    for group in groups
        sorted_pos = sort(group.tz)
        n = length(sorted_pos)
        
        lower_bound = sorted_pos[ceil(Int, n/3)]
        upper_bound = sorted_pos[ceil(Int, 2*n/3)]
        
        group_copy = copy(group)
        
        group_copy.relyaw = map(group_copy.tz) do val
            if val <= lower_bound
                return "right"
            elseif val > upper_bound
                return "left"
            else
                return "middle"
            end
        end
        append!(result, group_copy)
    end
    
    return result
end

function transfer_function_velo(allmoths, moth)
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

    
    stimpre = allmoths[moth]["velpre"]
    fxpre = allmoths[moth]["fxpre"]

    stimpost = allmoths[moth]["velpost"]
    fxpost = allmoths[moth]["fxpost"]

    fftprestim = (fft(float.(stimpre)))
    stimfreq = round.(fftfreq(length(stimpre),300),digits=1)
    fftpoststim = (fft(float.(stimpost)))

    fftfxpre = fft(float.(fxpre))
    fftfxpos = fft(float.(fxpost)) 
    fxfreq = round.(fftfreq(length(fxpre),1e4),digits=1)

    frset = Set(freqqs)

    stimidx= [i for i in 1:length(stimpre) if stimfreq[i] in frset]
    fxidx= [i for i in 1:length(fxpre) if fxfreq[i] in frset]

    sfxpre = fftfxpre[fxidx]
    sfxpost = fftfxpos[fxidx]

    sstimpre = fftprestim[stimidx]
    sstimpost = fftpoststim[stimidx]

    pre = sfxpre ./ sstimpre
    post = sfxpost ./ sstimpost
    return pre,post
end

## 
function transfer_function_coherence(allmoths,moth;axis="fx")
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]

    ft_ref = Dict(
        "fx" => 1,
        "fy" => 2,
        "fz" => 3,
        "tx" => 4,
        "ty" => 5,
        "tz" => 6
    )
    
    stimpre = allmoths[moth]["stimpre"]
    fxpre = allmoths[moth]["ftpre"][:,ft_ref[axis]]

    stimpost = allmoths[moth]["stimpost"]
    fxpost = allmoths[moth]["ftpos"][:,ft_ref[axis]]

    vpre = allmoths[moth]["velpre"]
    vpost = allmoths[moth]["velpost"]

    itp1 = interpolate(stimpre, BSpline(Linear()))
    stimpre_l = itp1(LinRange(1, length(itp1), Int(1e5)))

    itp2 = interpolate(stimpost, BSpline(Linear()))
    stimpost_l = itp2(LinRange(1, length(itp2), Int(1e5)))
    
    itp3 = interpolate(vpre, BSpline(Linear()))
    vpre_l = itp3(LinRange(1, length(itp3), Int(1e5)))
    
    itp4 = interpolate(vpost, BSpline(Linear()))
    vpost_l = itp4(LinRange(1, length(itp4), Int(1e5)))
    
    stimfreq = round.(fftfreq(100000,10000),digits=4)
    stimidx= [i for i in 1:length(stimpre) if stimfreq[i] in freqqs]


    coh_pre_pos = mt_coherence(hcat(fxpre,stimpre_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]
    coh_post_pos = mt_coherence(hcat(fxpost,stimpost_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]
    coh_pre_vel = mt_coherence(hcat(fxpre,vpre_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]
    coh_post_vel = mt_coherence(hcat(fxpost,vpost_l)';fs=1e4,nfft=100000).coherence[1,2,stimidx]


    return coh_pre_pos,coh_post_pos,coh_pre_vel,coh_post_vel
end
##
function get_mean_changes_yaw(allmoths)
    
    fs = 1e4
    moths = collect(keys(allmoths))
    freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70]
    peaks = DataFrame()
    all_data = DataFrame()
    for moth in moths 
        d = allmoths[moth]["data"]
        notpc = filter(col -> !contains(string(col), "_pc"), names(d))
        sel = filter(x -> !(x in ["wbtime","pos","vel"]), notpc)
        all_data=vcat(all_data,d[!,sel])
    end
    for moth in moths 
        println(moth)
        pre = allmoths[moth]["ftpre"][:,6]
        post = allmoths[moth]["ftpos"][:,6]

        fftpre = abs.(fft(pre)[2:50000])
        fftpost = abs.(fft(post)[2:50000])
        freqrange = round.(fftfreq(Int(1e5),fs)[2:50000],digits=4)
        d4t = all_data[all_data.moth.==moth,:]
        for f in freqqs
            id = findfirst(x -> x == f, freqrange)

            peakpre = fftpre[id]
            peakpost = fftpost[id]
            prdic = Dict("moth"=>moth,"freq" => f, "trial" => "pre", 
                "peak" => peakpre, "fz" => mean(d4t[d4t.trial.=="pre",:fz]) )
            podic = Dict("moth"=>moth,"freq" => f, "trial" => "post", 
                "peak" => peakpost,"fz" => mean(d4t[d4t.trial.=="post",:fz]) )
            push!(peaks,prdic,cols=:union)
            push!(peaks,podic,cols=:union)
        end
    end
    peaks.fz = peaks.fz .* -1 
    ##
    ##
    g=9.81
    ms = Dict(
        "2024_11_01" => Dict("pre"=>1.912,"post"=>2.075),
        "2024_11_04" => Dict("pre"=>2.149,"post"=>2.289),
        "2024_11_05" => Dict("pre"=>2.190,"post"=>2.592),
        "2024_11_07" => Dict("pre"=>1.801,"post"=>1.882),
        "2024_11_08" => Dict("pre"=>2.047,"post"=>2.369),
        "2024_11_11" => Dict("pre"=>1.810,"post"=>2.090),
        "2024_11_20" => Dict("pre"=>1.512,"post"=>1.784),
        "2025_01_30" => Dict("pre"=>2.13, "post"=>2.546),
        "2025_03_20" => Dict("pre"=>2.246, "post"=>2.344),
        "2025_01_14" => Dict("pre" => 2.07, "post"=> 2.531),
        "2025_04_02" => Dict("pre" => 1.7, "post"=> 1.916)
      )
    
    ##
    function normalize_fz(row, ms, g)
        return row.fz / ((ms[row.moth]["pre"]/1000) * g)
    end

    peaks.norm_fz = map(row -> normalize_fz(row, ms, g), eachrow(peaks))

    ##
    grouped = groupby(peaks,[:moth,:freq])

    changes = combine(grouped) do gdf 
        pre_vals = filter(r -> r.trial == "pre",gdf)
        post_vals = filter(r -> r.trial == "post",gdf)
        
        (
            fz_change =  (mean(post_vals.norm_fz) - mean(pre_vals.norm_fz)) / abs(mean(pre_vals.norm_fz)),
            gain_change = (mean(post_vals.peak) - mean(pre_vals.peak))/abs(mean(pre_vals.peak))*100
        )
    end

    mean_changes = combine(groupby(changes, :moth),
    :fz_change => mean => :mean_fz,
    :gain_change => mean => :mean_gain
    )
    mean_changes.mass .= 0.
    for row in eachrow(mean_changes)
        row.mass = ms[row.moth]["post"] - ms[row.moth]["pre"] 
    end
    ##
    return(mean_changes)
end