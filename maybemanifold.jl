using MultivariateStats
using CSV
using DataFrames
using GLMakie
using LinearAlgebra
using StatsBase
##

df = CSV.read("cachemoth1.csv",DataFrame)

##
ftnames = ["fx","fy","fz","tx","ty","tz"]

fts =  transpose(Matrix(df[!,ftnames]))

##

f = fit(PCA,fts)
Score = predict(f,fts)
pc1_score = Score[1,:]
##
lines(df.time_abs,pc1_score)
