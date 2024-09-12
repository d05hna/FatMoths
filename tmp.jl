oldcache = DataFrame()
startimes = Dict(
    "2024_06_06" => [0,50],
    "2024_06_20" => [0,17],
    "2024_06_24" => [0,16],
    "2024_07_09"  => [3,15] # use 001 for post
)
datadir = joinpath("/home","doshna","Documents","PHD","data","fatties")

##
old = CSV.read("mungedmusclesgoodtrackers.csv",DataFrame)
##
m = "2024_07_09"
##

tmp = read_ind(datadir,m,"hilbert")
tmp.time_abs = tmp.time_abs .+60
##
p = tmp[tmp.trial.==0,:]
##
df_filtered = p[(p.time_abs .>3 .&& p.time_abs .< 13), :]
# df_filtered.trial = ifelse.(df_filtered.time_abs .< 10, "pre", "post")
df_filtered.trial .= "pre"
##

cache = vcat(cache,df_filtered)