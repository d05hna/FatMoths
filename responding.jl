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
include("/home/doshna/Documents/PHD/comparativeMPanalysis/functions.jl")
##
datadir = "/home/doshna/Documents/PHD/data/fatties/"

files = glob("*/*.h5",datadir)
ps = glob("*/*.csv",datadir)
dpath = files[1]
spath = ps[1]
##

stim = CSV.read(spath,DataFrame)
##
function h5_to_df(path)
    h5open(path, "r") do file
        # Read the matrix from the file
        matrix = read(file, "data")
        
        # Read the column names from the file
        column_names = vec(read(file, "names"))
        
        # Create a DataFrame using the matrix and column names
        df = DataFrame(matrix, column_names)

        mapss = Dict(
            "ch1" => "fx",
            "ch2" => "fy",
            "ch3" => "fz",
            "ch4" => "tx",
            "ch5" => "ty",
            "ch6" => "tz" )
        if "ch1" in names(df)
            rename!(df,mapss)
        end

            
        return df
        end
    end
##
df = h5_to_df(dpath)

##
en = stim.Time[end]
beg = stim.Time[end]-20 

thestim = stim[stim.Time.>beg,:]
##
ftnames = ["fx","fy","fz","tx","ty","tz"]
##
itp = interpolate(thestim.Position,BSpline(Linear()))
            
full_pos = itp(LinRange(1, length(itp), size(df)[1]))
##
true_ft = DataFrame(transform_FT(transpose(Matrix(df[!,ftnames]))),ftnames)
##

function remove_bias(moth,datadir)
    params = Dict{String, Any}()

    q = h5_to_df(glob("$(moth)/*quiet.h5",datadir)[1])
    empty = h5_to_df(glob("$(moth)/*empty.h5",datadir)[1])
    ft_names = ["fx","fy","fz","tx","ty","tz"]

    bias = mean(Matrix(empty[!,ft_names]), dims=1)
    quiet = mean(Matrix(q[!,ft_names]), dims=1)

    quiet = transform_FT(transpose(quiet .- bias))

    A = [0 -quiet[3] quiet[2];
    quiet[3] 0 -quiet[1];
    -quiet[2] quiet[1] 0]
    B = -quiet[4:6]

    func(x) = norm(A*x-B)
    sol = optimize(func, [-10.0,-20.0,50.0], [10.0,20.0,60.0],[0.0,0.0,55.0])
    COM = Optim.minimizer(sol) # (x, y, z coordinates of most likely center of mass)
    M_transform = [1 0 0 0 0 0;
                0 1 0 0 0 0;
                0 0 1 0 0 0;
                0 COM[3] -COM[2] 1 0 0;
                -COM[3] 0 COM[1] 0 1 0;
                COM[2] -COM[1] 0 0 0 1]
    params["mass"] = quiet[3] / 9.81 * 1000 # N to g
    params["COM"] = COM
    params["M_transform"] = M_transform
    params["bias"] = bias
    return(params)
end

##

y = remove_bias("2024_06_04",datadir)

