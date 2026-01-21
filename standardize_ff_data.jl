using HDF5
using DataFrames 
using MAT 
using JLD2 
using FFTW 
using StatsBase
freqqs =  [0.2000,0.3000,0.5000,0.7000,1.100,1.300,1.700,1.900,2.300,2.900,3.700,4.300,5.300,6.100,7.900,8.900,11.30,13.70]

@load "fat_moth_free_flight.jld"
di = matread("Steven/TimeVarMass/moths_v73.mat")
cilat = di["Cilat"]
cflat = di["Cflat"]
##

FreeFlight

All_Data = Dict()
N = 2000
fs = 100
fr = round.(fftfreq(N,fs),digits=4)
stimidx= [i for i in 1:N if fr[i] in freqqs]

for (i, m) in enumerate(collect(keys(FreeFlight)))
    All_Data[m] = Dict() 
    All_Data[m]["Flower_Low"] = (FreeFlight[m]["flower_pre"] .- mean(FreeFlight[m]["flower_pre"])) ./ 5.74
    All_Data[m]["Flower_High"] = (FreeFlight[m]["flower_post"] .- mean(FreeFlight[m]["flower_post"])) ./ 5.74 
    All_Data[m]["Moth_Low"] = (FreeFlight[m]["moth_pre"] .- mean(FreeFlight[m]["flower_pre"])) ./ 5.74
    All_Data[m]["Moth_High"] = (FreeFlight[m]["moth_post"] .- mean(FreeFlight[m]["flower_post"])) ./ 5.74

    Hi = fft(All_Data[m]["Moth_Low"])
    Hf = fft(All_Data[m]["Moth_High"])

    All_Data[m]["Hi"] = Hi[stimidx]
    All_Data[m]["Hf"] = Hf[stimidx]

    All_Data[m]["Ci"] = cilat[:,i]
    All_Data[m]["Cf"] = cflat[:,i]

end


h5open("Steven/Final_Free_Flight.h5", "w") do file
    for (outer_key, inner_dict) in All_Data
        grp = create_group(file, outer_key)

        for (inner_key, value) in inner_dict
            grp[inner_key] = value
        end
    end
end
