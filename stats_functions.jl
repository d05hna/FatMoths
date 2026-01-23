function tf_stats(x:: Vector{ComplexF64}) 
    """
    METHODS TAKEN FROM ROTH et AL PNAS 2016  
    
    Compute the mean and std for gain and phase of a complex transfer function x 
    Takes in a vector of complex numbers 
    Here Used as a set of responses to a given driving frequencey
    mean and std of gain in log space exponentiated back 
    Circular mean and std for phase 

    NOTE : while mg +- std g is not symmetric because log space, p_std is symmetric around mp


    returns : mg, glow, ghigh, mp, p_std
    """ 

    # log mean of the gain 
    l_m_G = mean(log.(norm.(x)))

    # log std of the gain 
    l_std_g = sqrt(1/(length(x)-1) * sum((log.(norm.(x)) .- l_m_G).^2))

    # add and subtract std and then exponentiate
    high = exp(l_m_G + l_std_g)
    low = exp(l_m_G - l_std_g)
    # exponentiate the mean back 
    mg = exp(l_m_G)

    # circular mean of the phase 
    mp = angle(mean(exp.(angle.(x) .* im )))

    # Vector strength 
    R = norm(1/length(x) * sum(exp.(angle.(x) .* im )))

    # circular std of the phase 
    std_p = sqrt(-2 * log(R))

    return mg, low, high, mp, std_p
end


function fisher_g_pvalue(g, n)
    """
    Calculate the p-value for Fisher's g test.
    g: observed g statistic
    n: number of frequencies tested
    using log gamma distribution to avoid overflow issues
    """

    pval = 0.0
    upper_limit = floor(Int, 1/g)
    
    for k in 1:upper_limit

        term_log = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + (n - 1) * log(1 - k * g)
        term = exp(term_log)
        
        if k % 2 == 1
            pval += term
        else
            pval -= term
        end
    end
    return pval
end

function fisher_test_tracking(x,fs,f; win = 10 )
    """ 
    Fisher's g test for significant tracking outlined in Sponberg Sciecne 2015
    x -> time series of moth position 
    fs -> sampling frequency 
    f -> target driving frequency 
    win = 10 -> window size around the driving frequency to consider (+- win samples)
        frequency band is win * resolution of periodogram 
    """

    pxx = periodogram(x,fs=fs) 
    freqs = round.(pxx.freq, digits=2)

    id = findfirst(x -> x == f, freqs) 
    band = pxx.power[id-Int(win):id+Int(win)]
    pow = pxx.power[id] 
    tot = sum(band) 
    g = pow / tot 
    pval = fisher_g_pvalue(g,length(band))
    return pval
end 

function get_all_tf_stats(low,high,freqs; freq_max=20)
    """
    take in a low mass and high mass matrix of complex transfer functions
    rows are frequencies
    columns are moths
    freq_max -> maximum frequency to consider
    returns two dataframes of stats for low and high mass conditions
    """
    freqs = freqs[freqs .<= freq_max]
    pre = DataFrame()
    post = DataFrame()
    for (i,f) in enumerate(freqs)
        mg,glow,ghigh,mp,p_std = tf_stats(low[i,:])
        tmp = Dict(
            "freq"  => f,
            "mg"    => mg,
            "glow"  => glow,
            "ghigh" => ghigh,
            "mp"    => mp,
            "stp"  => p_std,
        )
        push!(pre, tmp,cols=:union)
        mg,glow,ghigh,mp,p_std = tf_stats(high[i,:])
        tmp = Dict(
            "freq"  => f,
            "mg"    => mg,
            "glow"  => glow,
            "ghigh" => ghigh,
            "mp"    => mp,
            "stp"  => p_std,
        )
        push!(post, tmp,cols=:union)
    end
    pre.mp = unwrap(pre.mp,circular_dims=true)
    post.mp = unwrap(post.mp,circular_dims=true)
    pre.glow = pre.mg .- (pre.mg .- pre.glow)./sqrt(size(low,2))
    pre.ghigh = pre.mg .+ (pre.ghigh .- pre.mg)./sqrt(size(low,2))
    post.glow = post.mg .- (post.mg .- post.glow)./sqrt(size(high,2))
    post.ghigh = post.mg .+ (post.ghigh .- post.mg)./sqrt(size(high,2))
    pre.stp = pre.stp ./ sqrt(size(low,2))
    post.stp = post.stp ./ sqrt(size(high,2))
    return pre, post
end 

