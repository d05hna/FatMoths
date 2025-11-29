using MAT
using CSV
using JLD2 
using FFTW
using GLMakie 
using DataFrames
using ColorSchemes
GLMakie.activate!()
theme = theme_minimal()
theme.palette = (; color = ColorSchemes.tab20)
# theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
set_theme!(theme)

@load "fat_moth_free_flight.jld" FreeFlight
@load "fat_moths_set_1.jld" allmoths

fs_closed = 100 
fs_open = 300 
N_closed = 2000 
N_open = 3000 
freqqs = [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]

##
errors = Dict() 
for moth in collect(keys(FreeFlight))
    d = FreeFlight[moth]
    ms = d["moth_pre"]
    fs = d["flower_pre"]
    es = ms .- fs 
    fes = fft(es) 
    mf = d["moth_post"]
    ff = d["flower_post"]
    ef = mf .- fs
    fef = fft(ef) 
    errors[moth] = Dict("pre"=> es,"post"=> ef,"fpre" => fes,"fpost"=>fef)
end 
fr_closed = fftfreq(N_closed,fs_closed)
fr_open = fftfreq(N_open,fs_open)[1:201]
## Start the Figure with the Closed Loop 
F = Figure(size=(800,800))
ax = Axis(F[1,1],xscale=log10,title="Closed Loop Error",ylabel="Magnitude",xlabel="Frequency",
    limits = (nothing,nothing,0,50))
vlines!(ax,freqqs,color=:grey,alpha=0.5,linestyle=:dash)

for m in collect(keys(errors))
    lines!(ax,fr_closed[1:401],abs.(errors[m]["fpre"])[1:401] ./ N_closed)
    lines!(ax, fr_closed[1:401],abs.(errors[m]["fpost"])[1:401]./N_closed)
end

ax2 = Axis(F[1,2],xscale=log10,title="Open Loop Flower",ylabel = "Magnitude",xlabel="Frequency",
    limits = (nothing,nothing,0,50))
vlines!(ax2,freqqs,color=:grey,alpha=0.5,linestyle=:dash)
for m in collect(keys(allmoths))
    fpre = abs.(fft(allmoths[m]["stimpre"]))[1:201] ./ N_open 
    fpost = abs.(fft(allmoths[m]["stimpost"]))[1:201] ./ N_open 
    lines!(ax2,fr_open,fpre)
    lines!(ax2,fr_open,fpost)
end

F
""" 
YOU NEED TO ENSURE UNITS BC PIXELS COULD BE DECEVIING 
GO AND LOOK FOR HIS UNIT CONVERSION AND TURN OL TO Same Unit you have the conversiton 
"""
##