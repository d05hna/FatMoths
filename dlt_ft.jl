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
##
d = CSV.read("DLTdv8_data_000_0xypts.csv",DataFrame)
##
x_pos = d.pt1_cam1_X
y_pos = d.pt1_cam1_Y
time = range(0,10,3000)
## Make sure it is only X direction I need to care about
fig = Figure()
ax1 = Axis(fig[1,1],limits=(0,10,0,800))
ax2 = Axis(fig[2,1],limits=(0,10,0,800))
lines!(ax1,time,x_pos)
ax1.title="X Cord"
ax1.xlabel="Time"
lines!(ax2,time,y_pos)
ax2.title = "Y Cord"
ax2.xlabel = "Time"
save("exampleflowertrack.png",fig,px_per_unit=4)
fig
## Fourrier Transform

fft_x = fft(x_pos)
fs = 300 
freqs = fftfreq(length(x_pos),fs)
power = abs.(fft_x)

pos_freqs = freqs[2:Int(length(x_pos)/2)]
pos_power = freqs[2:Int(length(x_pos)/2)]

##
f = Figure()
ax1 = Axis(f[1,1])
lines!(ax1,time,x_pos)
ax1.title="X Cord"
ax1.xlabel="Time"
ax2 = Axis(f[2,1],xscale=log10)
lines!(ax2,pos_freqs,pos_power)
ax2.title="Fourier"
ax2.xlabel="Frequency"
ax2.ylabel="power"
f