using CSV, DataFrames,GLMakie,AlgebraOfGraphics
using GLMakie
GLMakie.activate!()
sicbTheme = theme_minimal()
sicbTheme.textcolor=:black
sicbTheme.ytickfontcolor=:black
sicbTheme.fontsize= 20
sicbTheme.gridcolor=:lightcyan
sicbTheme.gridalpga=0.5
sicbTheme.palette = (color = [:steelblue,:firebrick],)
set_theme!(sicbTheme)
##

df = CSV.read("MID.csv",DataFrame)
moths_ids = ["Moth $(i)" for i in 1:length(unique(df.moth))]
mothref = DataFrame(:moth => sort(unique(df.moth)), :moth_id => moths_ids)
df = leftjoin(df,mothref,on=:moth)

##
plot = data(df)*mapping(:axis => renamer(["fx" => "Side Slip","tz"=>"Yaw","ty" => "Roll"]) => "Axis" ,:value_abs => "Projection Weight",
    dodge=:condition,color=:condition => renamer(["LOW"=>"Low Mass","HIGH"=>"High Mass"])=>"Body Condition",layout=:moth_id)*visual(BarPlot) 

f = draw(plot,figure=(; size=(1000,1000)),legend=(; position=:top),axis=(; xlabelsize=30,ylabelsize=30,xticklabelsize=30))
save("GoodFigs/trackingstrategies.png",f)

