using CSV
using DataFrames
using AlgebraOfGraphics
using GLMakie
using StatsBase
using HypothesisTests
using PrettyTables
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:turquoise,:coral],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)
##

df = CSV.read("fullhighdimTIME.csv", DataFrame)
for col in names(df)
    df[!, col] = replace(df[!, col], "NA" => missing)
end
for col in names(df)
    try
        df[!, col] = [ismissing(x) ? missing : parse(Float64, x) for x in df[!, col]]
    catch

    end
end
##

firstphase = filter(str -> !in(str[end], ['t']), names(df))
firstspike = select(df,firstphase)
select!(firstspike,["moth","wb","trial","fz","wbfreq","lax1","lba1","lsa1","ldvm1","ldlm1","rdlm1","rdvm1","rsa1","rba1","rax1"])
rename!(firstspike,["moth","wb","trial","fz","wbfreq","lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"])

##
pairs = [["ldlm","rdlm"],["ldlm","ldvm"],["rdlm","rdvm"],["lax","ldlm"],["lba","ldlm"],["lsa","ldlm"]]
tits = ["dlm","leftpower","rightpower","lax","lba","lsa"]

for (i,p) in enumerate(pairs)
    tmp = dropmissing(select(firstspike,["moth","trial",p[1],p[2]]))
    tmp.diff = tmp[!,p[1]] - tmp[!,p[2]]

    plot = data(tmp) * mapping(:diff => "Timing Offset",row = :moth,color=:trial => 
        renamer(["pre"=>"Lower Mass","post"=>"Higher Mass"])) * histogram(bins=100,normalization=:probability) * visual(alpha=0.5)
    f = draw(plot,figure = (;resolution=(600,1200)),axis = (; title="$(tits[i]) Offset "))
    save("FatMothSet1/offsets/$(tits[i]).png",f,px_per_unit=4)
end
    
## What about Power Offsets by Z force increase 

powers = select(df,:moth,:trial,:fz,:ldlm1,:rdlm1,:ldvm1,:rdvm1)

dropmissing!(powers)

powers.leftdif = powers.ldlm1 - powers.ldvm1
powers.rightdif = powers.rdlm1 - powers.rdvm1
powers.dlmdif = powers.rdlm1 - powers.ldlm1

plot = data(powers) * mapping(:dlmdif,color=:trial => 
    renamer(["pre"=>"Lower Mass","post"=>"Higher Mass"]))*histogram(bins=100,normalization=:probability) |> draw

## OKAY LOOK AT THE VARIANCES OF THE PRE AND POST DISTRIBUTIONS. PLOT VARIANCES AS A FUNCTION OF MEAN Z FORCE FOR THAT TRIAL AND THEN COLOR ON THE PRE AND POST 
levenes = DataFrame()

for moth in unique(powers.moth)
    subdf = @view powers[powers.moth.==moth,:]

    dlmpre = subdf[subdf.trial.=="pre",:dlmdif]
    dlmpost = subdf[subdf.trial.=="post",:dlmdif]

    rpre = subdf[subdf.trial.=="pre",:rightdif]
    rpost = subdf[subdf.trial.=="post",:rightdif]

    lpre = subdf[subdf.trial.=="pre",:leftdif]
    lpost = subdf[subdf.trial.=="post",:leftdif]

    pdlm = pvalue(LeveneTest(dlmpre,dlmpost))
    pr = pvalue(LeveneTest(rpre,rpost))
    pl = pvalue(LeveneTest(lpre,lpost))

    tmp = Dict(
        "moth" => moth,
        "DLM" => pdlm,
        "RightPower" => pr,
        "LeftPower" => pl
    )
    push!(levenes,tmp,cols=:union)
end
select!(levenes,:moth,:)
## print it as a pretty table and then highlight ones that are not significant for Lenes 
hl_greater = Highlighter(
    (data, i, j) -> (j in [2,3,4]) && data[i,j] > 0.05,  # condition for columns 2 and 3
    bold = true,
    foreground = :green
)
pretty_table(levenes,highlighters=hl_greater)
##

powerdiff = select(powers,:moth,:trial,:fz,:leftdif,:rightdif,:dlmdif)

save("powerdiffs.csv",powerdiff)