using DataFrames,CSV,StatsBase,HypothesisTests,GLM
##
df = CSV.read("powermusclestimings.csv",DataFrame)
##

m1 = df[df.moth.=="2024_11_01",:]

