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
using GLM
using Colors
using KernelDensity
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
include("me_functions.jl")
##
paired_10_colors = [
    colorant"#A6CEE3",  
    colorant"#FB9A99", colorant"#6A3D9A",
    colorant"#1F78B4", colorant"#B2DF8A", 
    colorant"#33A02C", 
     colorant"#E31A1C", 
    colorant"#FDBF6F", colorant"#FF7F00", 
    colorant"#CAB2D6"
]
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
# theme.pallette = (color = paired_10_colors,)
theme.palette = (color = [:turquoise,:coral],)

set_theme!(theme)
##
function calculate_fft(data, fs)
    fft_data = fft(data)[2:end]
    fr_data = fftfreq(length(data), fs)[2:end]
    return fr_data, fft_data
end

@load "munged_all_axisv2.jld" all_axis
## first lets just plot the z force I want to see how it changes. 

fig = Figure(resolution=(1200,1200))
axes = []

time = range(0,step=1/10000,length=100000)

for (i,(key,d)) in enumerate(all_axis)
    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1
    ax = Axis(fig[row, col], title = key)
    push!(axes, ax)
    
    lines!(ax,time,d["pre"].fz .* 1000,label="Before Feeding",alpha=0.5)
    lines!(ax,time,d["post"].fz .*1000,label="After Feeding",alpha=0.5)
    ax.xlabel="Time"
    ax.ylabel = "Z Force (mN)"

end

linkyaxes!(axes...)
linkxaxes!(axes...)

Label(fig[0, :], "Comparison of Pre and Post fz Values", fontsize = 24)

# Add a legend
Legend(fig[1:3, 3], axes[1], "Data", framevisible = false)

display(fig)

# save("allgoodmoths/ZFORCEtrace.png",fig,px_per_unit=4)

## Lets get the fft for each moth on each axis and then save them all 

freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]
fs = 10000
allpeaks= DataFrame()
ftnames = ["fx","fy","fz","tx","ty","tz"]
for ft in ftnames
    for (moth,dic) in all_axis
        pre = dic["pre"][!,ft]
        post = dic["post"][!,ft]

        frpre, fftpre = calculate_fft(pre,fs)
        frpost, fftpost = calculate_fft(post,fs)

        idxs = frpre .>0 .&& frpre .<20
        fridx = findall(x->x in freqqs,round.(frpre,digits=2))

        peakspre = abs.(fftpre[fridx])
        peakspost = abs.(fftpost[fridx])
        if ft == "fx"
            tmp = DataFrame(
                freq = freqqs,
                pre = peakspre,
                post = peakspost,
                change = peakspost ./ peakspre
            )
            tmp.moth .= moth
            allpeaks = vcat(allpeaks,tmp,cols=:union)
        end
        fig = Figure(size=(800, 600), fontsize=14)
        ax = Axis(fig[1, 1], 
            xlabel = "Frequency", 
            ylabel = "Amplitude",
            title = "$moth $(ft) Frequency Response",
            xscale = log10
        )
        lines!(ax,frpre[idxs],abs.(fftpre[idxs]),label="Before Feeding",alpha=0.5)
        lines!(ax,frpost[idxs],abs.(fftpost[idxs]),label="After Feeding",alpha=0.5)
        axislegend(ax, "Trial", position=:rt)
        vlines!(ax,freqqs,color=:grey,linestyle=:dash,alpha=0.3)

        # save("allgoodmoths/ffts/$(ft)/$(moth).png",fig,px_per_unit=4)


    end
end
##
avg_change = DataFrame()
for m in keys(all_axis)
    d = mean(allpeaks[allpeaks.moth.==m,:].change)
    tmp = Dict("moth"=> m,"gain"=>d)

    pre = mean(all_axis[m]["pre"].tz)
    post = mean(all_axis[m]["post"].tz)

    tmp["fz"] = post-pre

    push!(avg_change,tmp,cols=:union)

end

## lets do wb and then mung the moths and trials together for mean axis 

function get_wb_means(nested_dict,moth,trial,maxwb)
    df = nested_dict[moth][trial]
    fs = 10000
            
    z_bandpass = [5, 30]
    ft_lowpass = 1000

    cheby_bandpass = digitalfilter(Bandpass(z_bandpass...; fs=fs), Chebyshev1(4, 4))


    df.moth .= moth
    df.trial .= trial

    df.wb = @pipe filtfilt(cheby_bandpass, df.fz) |>
        hilbert .|>
        angle .|> 
        sign |>
        (x -> pushfirst!(diff(x), 0)) |> # diff reduces len by 1, pad zero at beginning
        cumsum(_ .> 0.1)

    df = @pipe df |>
        groupby(_, :wb) |> 
            transform(_, :wb => (x -> length(x) / fs) => :wblen) |> 
            #     # Mark wingbeats to remove 
            #     # (using bool column instead of removing now to allow proper spike unwrapping)
            #     # Remove wingbeats if too short or too long 
                groupby(_, :wb) |> 
                    @transform(_, :validwb = (first(:wblen) >= 1/30) .& (first(:wblen) <= 1/16))

    df = @pipe df |> 
        groupby(_, :wb) |> 
        transform(_, Symbol.(ftnames) .=> mean .=> Symbol.(ftnames))
        

    unique!(df)
    final = df[df.validwb,:]

    final[!,ftnames] = Matrix(final[!,ftnames]) 
    final.wb = final.wb .+ maxwb
    final.wbfreq = 1 ./final.wblen
    return(final)
end
##
full_means = DataFrame()
maxwb = 0
for moth in keys(all_axis)
    for trial in ["pre","post"]
        d = get_wb_means(all_axis,moth,trial,maxwb)
        full_means = vcat(full_means,d,cols=:union)
        maxwb = maximum(full_means.wb)
    end
end
##
"""
First looking at all the frequencys. 
How does the change in Gain on avergae across all frequencies
reflect the effort that the moth is trying to support this new weight? 
The More negative the FZ becomes, the more weight is supported. 

My Hypothesis is that the gain change will be more apparent the more weight being supported
"""
mean_fz = combine(groupby(full_means, [:moth, :trial]), :fz => mean => :mean_fz)
wide_fz = unstack(mean_fz, :trial, :mean_fz)
transform!(wide_fz,
    [:pre, :post] => ByRow((pre, post) -> -1*(post - pre) / abs(pre) *100 ) => :change_fz)

# Round the percent change to two decimal places
transform!(wide_fz, :change_fz => ByRow(x -> round(x, digits=2)) => :percent_change)

big_avg = leftjoin(wide_fz[!,[:moth,:change_fz]],avg_change[!,[:moth,:gain]],on=:moth)

##

fig = Figure()
ax = Axis(fig[1,1],xlabel = "% Increase in Mean WB Z Force",ylabel="Gain Change Multiple", title = "All Frequencies",
    limits = (nothing,nothing,nothing,maximum(big_avg.gain)+1))
scatter!(ax, big_avg.change_fz, big_avg.gain,color=:turquoise,markersize=15)

model = lm(@formula(gain ~change_fz ), big_avg)

rsq = r2(model)

p = coeftable(model).cols[4][2]

x_range = range(minimum(big_avg.change_fz), maximum(big_avg.change_fz), length=100)
y_pred = predict(model, DataFrame(change_fz=x_range)) 

lines!(ax, x_range, y_pred, color=:coral)

text!(ax, maximum(big_avg.change_fz), maximum(big_avg.gain)+0.5, 
    text="R² = $(round(rsq, digits=3))\np = $(round(p, digits=3))",
align=(:right, :bottom))

display(fig)
save("allgoodmoths/gainfz/Allfreqsgainfz.png",fig,px_per_unit=4)

##
peaksbyfreq = leftjoin(allpeaks[!,[:moth,:freq,:change]],wide_fz[!,[:moth,:change_fz]],on=:moth)

for f in peaksbyfreq.freq
    tmp = @view peaksbyfreq[peaksbyfreq.freq .== f,:]
    fig = Figure()
    ax = Axis(fig[1,1],xlabel = "Wing Stroke Z Force Change Multiple",ylabel="Gain Change Multiple", title = "$f Hz",
        limits = (nothing,nothing,nothing,maximum(tmp.change)+1))
    scatter!(ax, tmp.change_fz, tmp.change,color=:turquoise,markersize=15)

    save("allgoodmoths/gainfz/$f.png",fig,px_per_unit=4)
end
## Lets see if there are any trends in the muscle activities 

muscles = CSV.read("mungedmusclesgoodtrackers.csv",DataFrame)
##

for moth in unique(muscles.moth)
    plot = data(muscles[muscles.moth.==moth,:])*mapping(:phase,row=:muscle,color=:trial=>renamer("pre"=>"Before Feeding","post"=> "After Feeding")=>"Trial")*histogram(bins=100,normalization=:probability)*
        visual(alpha=0.7)
    fig = draw(plot,figure=(; resolution = (600,600)),axis=(; yticks=[0]))
    save("allgoodmoths/muscles/$moth.png",fig,px_per_unit=4)
end
## wait actually lets look at frequency

plot = data(muscles)*mapping(:wbfreq => "WB Frequency",color=:trial=>renamer("pre"=>"Before Feeding","post"=> "After Feeding")=>"Trial",layout=:moth)*histogram(bins=100,normalization=:probability)*visual(alpha=0.5) 
fig = draw(plot,axis=(; xticks=[15,20,25]))

save("allgoodmoths/wbfreq.png",fig,px_per_unit=4)
## muscles werent super clear but I dont have full muscles for all I have dlm - dvm for 2 of them lets check the offset

df_min_phase = combine(groupby(muscles, [:moth, :trial, :wb, :muscle,:fz]), :phase => minimum => :min_phase)
df_wide = unstack(df_min_phase, [:moth, :trial, :wb,:fz], :muscle, :min_phase)

select!(df_wide,:moth,:wb,:trial,:fz,:ldlm,:rdlm,:ldvm,:rdvm)
dropmissing!(df_wide)

##
df_wide.left = df_wide.ldlm - df_wide.ldvm
df_wide.right = df_wide.rdlm - df_wide.rdvm
select!(df_wide,:moth,:wb,:trial,:left,:right,:fz)

final = stack(df_wide, [:left, :right], 
                variable_name = :side, value_name = :diff)
##
plot = data(final)*mapping(:trial => renamer(["pre"=>"pre","post"=>"post"]),:diff,color=:trial=> renamer(["pre"=>"pre","post"=>"post"]),layout=:side)*visual(BoxPlot)|> draw

##
function create_custom_colormap(base_color, n_colors=100)
    base = parse(Colorant, base_color)
    colors = [RGBA(1,1,1,0)]  # Start with transparent white
    for i in 1:n_colors
        alpha = i/n_colors
        push!(colors, RGBA(base.r, base.g, base.b, alpha))
    end
    return cgrad(colors)
end
##
function make_dif_fz_plot(final)
    fig = Figure(resolution=(800,800))
    ms = unique(final.moth)
    sides = ["left","right"]

    base_colors = Dict("pre" => :blue, "post" => :red)
    colormaps = Dict("pre" => create_custom_colormap(:blue), 
                    "post" => create_custom_colormap(:red))

    for (i,moth) in enumerate(ms)
        for (j,side) in enumerate(sides)
            ax = Axis(fig[i,j],title = "$moth, $side",limits=(-1,0,0.02,0.07))
            sub = final[(final.moth.==moth) .&& (final.side.==side),:]
            for trial in ["pre","post"]
                trisub = sub[sub.trial.==trial,:]
                kern = kde((trisub.diff, trisub.fz))
                dens = kern.density ./ maximum(kern.density)
                threshold = 0.05  # Adjust this value to control the cutoff
                dens[dens .< threshold] .= 0
                
                contourf!(ax, kern.x, kern.y, dens, 
                        levels=20, colormap=colormaps[trial])

            
            end
            if i == length(ms)
                ax.xlabel = "DLM DVM Phase Offset"
            end
            if j == 1
                ax.ylabel = "Mean Wingstroke Z Force "
            end
        end
    end
    elements = [MarkerElement(color=base_colors[t], marker=:circle) for t in ["pre", "post"]]
    Legend(fig[1:length(ms), length(sides)+1], elements, ["pre", "post"], "Trial")

    return(fig)

end
##
p  = make_dif_fz_plot(final)
save("allgoodmoths/zforcephaseoffset.png",p,px_per_unit=3)