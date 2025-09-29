using CSV,GLMakie, DataFrames,Glob,AlgebraOfGraphics
##
datadir = "/media/doshna/My Passport/DOSHNA/FatMoths/Videos/"
fs = glob("*/*/*_timestamps.csv",datadir)
##
all_data = DataFrame()

for f in fs
    d = CSV.read(f,DataFrame)
    rename!(d, Symbol.(["num","time"]))

    off = [d.time[i] - d.time[i-1] for i in 2:nrow(d)]
    tmp = DataFrame(off = off)
    tmp.num = d.num[2:end]
    tmp.cam .= split(f,"/")[end-1]
    tmp.moth .= split(split(f,"/")[end],"_")[1]
    tmp.fps  = 1 ./tmp.off

    all_data = vcat(all_data,tmp,cols=:union)

end
##
all_data = all_data[all_data.moth .!= "20250320" .&& all_data.moth .!= "20250402",:]
all_data=all_data[all_data.fps .> 100,:]
plot = data(all_data)*mapping(:num, :fps,color=:cam,layout=:moth)|>draw

