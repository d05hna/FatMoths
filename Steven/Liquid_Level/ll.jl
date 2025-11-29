using CairoMakie,CSV,DataFrames,GLM
CairoMakie.activate!()
theme = theme_minimal()
# theme.palette = (; color = ColorSchemes.tab20)
theme.palette = (; color = [:firebrick,:steelblue])
theme.fontsize = 20
# theme.font = GLMakie.to_font("Dejavu")
set_theme!(theme)

pixel_conversion = 20 / 142 # mm over pixels height of rear cylinder of flower 


df = CSV.read("/home/doshna/Documents/PHD/FatMoths/Steven/Liquid_Level/llxypts.csv",DataFrame)

rename!(df,["x","y"])
df.t = range(0,200,length=nrow(df)) # the orriginal bout of feeding is 200 seconds long 
df.y .*= -1*pixel_conversion
df.y .-= maximum(df.y)
df.x .*= pixel_conversion
m = lm(@formula(y~t),df)
r = r2(m)
p = predict(m)

##
F = Figure(size=(600,600))
ax = Axis(F[1,1],xlabel="Time (s)",ylabel = "Sucrose Displacement (mm)")
scatter!(ax,df.t,df.y,color=:black,alpha=0.8,markersize=5)
lines!(ax,df.t,p,color=:firebrick,linewidth=4,linestyle=:dash)
text!(ax,150,-2,text=L"r^2=0.995",color=:firebrick)
save("Liquid_Level_Fig.svg",F,px_per_unit=4)
F