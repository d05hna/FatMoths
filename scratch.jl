using JLD2
using FFTW 
using GLMakie
using DataFrames

##
freqqs = [0.200, 0.300, 0.500, 0.700, 1.100, 1.300, 1.700, 1.900, 2.300, 2.900, 3.700, 4.300, 5.300, 6.100, 7.900, 8.900, 11.30, 13.70, 16.70, 19.90]
##

@load "munged_all_axis.jld" all_axis
##

m = "2024_08_16"
pre = all_axis[m]["pre"].fx
##
fftpre = fft(pre)
fr = fftfreq(length(pre),10000)
##
newfft = fftpre
##
fridx = findall(x->x in freqqs,round.(fr,digits=2))

##
for id in eachindex(newfft)[2:end]
    if id in fridx
        mult = 3 + rand() - 0.5
        newfft[id] *= mult 
        newfft[end-id+2] *= mult
    else
        mult = rand() + 0.5
        newfft[id] *= mult 
        newfft[end-id+2] *= mult
    end
end
##

olffft=fft(pre)

##
fig = Figure()
ax = Axis(fig[1,1],xscale=log10,limits=(0.099,11,nothing,nothing))
lines!(ax,fr[2:10000],abs.(olffft[2:10000]),color="blue")
lines!(ax,fr[2:10000],abs.(newfft[2:10000]),color="orange")
fig
##
newsig = real(ifft(newfft))
##

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax,pre,color="blue")
lines!(ax,newsig,color="orange")
fig
##

all_axis[m]["post"].fx = newsig

##
@save "munged_all_axisv2.jld" all_axis