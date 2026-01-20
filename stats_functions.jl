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
    band = pxx.power[id-Int(win/2):id+Int(win/2)]
    pow = pxx.power[id] 
    tot = sum(band) 
    g = pow / tot 
    pval = fisher_g_pvalue(g,win+1)
    return pval
end 