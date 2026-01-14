using JSON, DataFrames, FFTW, GLMakie,AlgebraOfGraphics,StatsBase,Pipe,DataFramesMeta, DSP
GLMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)
include("me_functions.jl")
fs = Int(1e4) 
N = Int(1e5)
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]


cvs = ["CV$i" for i in 0:4]
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties","vahid_results","cv_results","CVs")
##
all_data = Dict() 
for cv in cvs 
    all_data[cv] = Dict() 
    dat = JSON.parsefile(joinpath(datadir,cv,"exp_all_variational0","latents.json"))
    all_data[cv]["flower"] = reduce(hcat,dat["flower"])' |> x -> Float64.(x)

    for i in 0:6 
        moth = "moth_$i"
        all_data[cv][moth]=Dict()
        low_share = reduce(hcat,dat[moth]["low_share_latent"])' |> x -> Float64.(x)
        all_data[cv][moth]["low"] = low_share[:,1:3]
        all_data[cv][moth]["lowshare"] = low_share[:,4:6]
        high_share = reduce(hcat,dat[moth]["share_high_latent"])' |> x -> Float64.(x)
        all_data[cv][moth]["high"] = high_share[:,4:6]
        all_data[cv][moth]["highshare"]= high_share[:,1:3]
    end
end
##
function get_lat_tracking_fig(all_data,m)

    F = Figure(size=(1400,800))
    for i in 1:3
        ax = Axis(F[1,i],title="Low $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
        ax.yticks=[0.001,0.01,0.1,1]
        ax.limits=(nothing,nothing,0.0001,0.1)
        ax2 = Axis(F[2,i],ylabel="Phase",xscale=log10)
        ax2.xticks=[0.1,1,10]
        ax2.xticklabelsvisible=false
        ax2.limits=(0.1,nothing,nothing,nothing)
        linkxaxes!(ax2,ax)
        for c in cvs
            flower = all_data[c]["flower"]
            low = all_data[c][m]["low"]
            tf = tf_freq(flower[:,1],low[:,i],freqqs,fs)
            lines!(ax,freqqs,abs.(tf),color=:steelblue,linewidth=1)
            lines!(ax2,freqqs,unwrap_negative(angle.(tf)),color=:steelblue,linewidth=1)
        end
        
    end
    for i in 1:3
        ax = Axis(F[1:2,i+3],title="Share $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
        ax2 = Axis(F[3:4,i+3],ylabel="Phase",xscale=log10)

        ax.yticks=[0.001,0.01,0.1,1]
        ax.limits=(nothing,nothing,0.0001,0.1)
        ax2.xticks=[0.1,1,10]
        ax2.limits=(0.1,nothing,nothing,nothing)
        linkxaxes!(ax2,ax)

        for c in cvs 
            flower = all_data[c]["flower"]
            lowshare = all_data[c][m]["lowshare"]
            highshare = all_data[c][m]["highshare"]
                
            tf_low = tf_freq(flower[:,1],lowshare[:,i],freqqs,fs)
            tf_high = tf_freq(flower[:,1],highshare[:,i],freqqs,fs)

            lines!(ax,freqqs,abs.(tf_low),color=:steelblue,linewidth=1)
            lines!(ax,freqqs,abs.(tf_high),color=:firebrick,linewidth=1)

            lines!(ax2,freqqs,unwrap_negative(angle.(tf_low)),color=:steelblue,linewidth=1)
            lines!(ax2,freqqs,unwrap_negative(angle.(tf_high)),color=:firebrick,linewidth=1)
        end

    end
    for i in 1:3
        ax = Axis(F[3,i],title="High $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
        ax2 = Axis(F[4,i],xlabel="Freq",ylabel="Phase",xscale=log10)

        ax.yticks=[0.001,0.01,0.1,1]
        ax.limits=(nothing,nothing,0.0001,0.1)
        ax2.xticks=[0.1,1,10]
        ax2.limits=(0.1,nothing,nothing,nothing)
        linkxaxes!(ax2,ax)

        for c in cvs 
            flower = all_data[c]["flower"]
            high = all_data[c][m]["high"]
            tf = tf_freq(flower[:,1],high[:,i],freqqs,fs)
            lines!(ax,freqqs,abs.(tf),color=:firebrick,linewidth=1)
            lines!(ax2,freqqs,unwrap_negative(angle.(tf)),color=:firebrick,linewidth=1)
        end


    end

    return F 
end
##
for i in 0:6
    m = "moth_$i"
    f = get_lat_tracking_fig(all_data,m)
    save("Figs/VahidFigs/Lat_CVs/all_cvs_$m.png",f)
end
##
""" 
Okay lets look at mean responses across all moths for each cv trial 
"""
## 
mean_data = Dict() 
c = "CV1"
for c in cvs
    d = all_data[c]
    flower = d["flower"][:,1]
    low = zeros(18,7,3) |> x -> Complex.(x)
    high = zeros(18,7,3) |> x -> Complex.(x)
    sharelow = zeros(18,7,3) |> x -> Complex.(x)
    sharehigh = zeros(18,7,3) |> x -> Complex.(x)

    for i in 1:3
        for j in 1:7 
            low[:,j,i] = tf_freq(flower,d["moth_$(j-1)"]["low"][:,i],freqqs,fs)
            high[:,j,i] = tf_freq(flower,d["moth_$(j-1)"]["high"][:,i],freqqs,fs)
            sharelow[:,j,i] = tf_freq(flower,d["moth_$(j-1)"]["lowshare"][:,i],freqqs,fs)
            sharehigh[:,j,i] = tf_freq(flower,d["moth_$(j-1)"]["highshare"][:,i],freqqs,fs)
        end
    end
    low = dropdims(mean(low,dims=2);dims=2)
    high = dropdims(mean(high,dims=2);dims=2)
    lowshare = dropdims(mean(sharelow,dims=2);dims=2)
    highshare = dropdims(mean(sharehigh,dims=2);dims=2)
    mean_data[c] = Dict(
        "low"=> low,
        "high" => high,
        "lowshare"=> lowshare,
        "highshare" => highshare
    )
end
##
F = Figure(size=(1600,800))
blues = [:navyblue,:blue,:steelblue,:slateblue,:steelblue4]
for i in 1:3
    ax = Axis(F[1,i],title="Low $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
    ax.yticks=[0.001,0.01,0.1,1]
    ax.limits=(nothing,nothing,0.0001,0.1)
    ax2 = Axis(F[2,i],ylabel="Phase",xscale=log10)
    ax2.xticks=[0.1,1,10]
    ax2.xticklabelsvisible=false
    ax2.limits=(0.1,nothing,nothing,nothing)
    linkxaxes!(ax2,ax)
    for (k,c) in enumerate(cvs)
        tf = mean_data[c]["low"][:,i]
        lines!(ax,freqqs,abs.(tf),linewidth=2,color=blues[k])
        lines!(ax2,freqqs,unwrap_negative(angle.(tf)),linewidth=2,color=blues[k])
    end
end

save("Figs/VahidFigs/Lat_CVs/mean_lows.png",F)
F
##
F = Figure(size=(1600,800))
reds = [:salmon1,:tomato,:firebrick,:indianred,:darkred]

for i in 1:3
    ax = Axis(F[1,i],title="High $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
    ax.yticks=[0.001,0.01,0.1,1]
    ax.limits=(nothing,nothing,0.0001,0.1)
    ax2 = Axis(F[2,i],ylabel="Phase",xscale=log10)
    ax2.xticks=[0.1,1,10]
    ax2.xticklabelsvisible=false
    ax2.limits=(0.1,nothing,nothing,nothing)
    linkxaxes!(ax2,ax)
    for (k,c) in enumerate(cvs)
        tf = mean_data[c]["high"][:,i]
        lines!(ax,freqqs,abs.(tf),linewidth=2,color=reds[k])
        lines!(ax2,freqqs,unwrap_negative(angle.(tf)),linewidth=2,color=reds[k])
    end
end
save("Figs/VahidFigs/Lat_CVs/mean_highs.png",F)
F
##
F = Figure(size=(1600,800))
for i in 1:3
    ax = Axis(F[1,i],title="Share $i",xticklabelsvisible=false,ylabel="Gain",xscale=log10,yscale=log10)
    ax.yticks=[0.001,0.01,0.1,1]
    ax.limits=(nothing,nothing,0.0001,0.1)
    ax2 = Axis(F[2,i],ylabel="Phase",xscale=log10)
    ax2.xticks=[0.1,1,10]
    ax2.xticklabelsvisible=false
    ax2.limits=(0.1,nothing,nothing,nothing)
    linkxaxes!(ax2,ax)
    for (k,c) in enumerate(cvs)
        tf_low = mean_data[c]["lowshare"][:,i]
        tf_high = mean_data[c]["highshare"][:,i]

        lines!(ax,freqqs,abs.(tf_low),linewidth=2,color=blues[k])
        lines!(ax2,freqqs,unwrap_negative(angle.(tf_low)),linewidth=2,color=blues[k])

        lines!(ax,freqqs,abs.(tf_high),linewidth=2,color=reds[k])
        lines!(ax2,freqqs,unwrap_negative(angle.(tf_high)),linewidth=2,color=reds[k])
    end
end
save("Figs/VahidFigs/Lat_CVs/mean_shared.png",F)
F