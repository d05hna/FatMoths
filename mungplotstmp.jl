function quartiles(y)
    q1 = quantile(y, 0.25)
    median = quantile(y, 0.5)
    q3 = quantile(y, 0.75)
    return (q1 = q1, median = median, q3 = q3)
end
##
# Group the data by frequency and trial
grouped_data = combine(groupby(all_moths, [:freq, :trial]), :gain => quartiles => AsTable)
##
fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Log10 (Newtons / Position)",
    title = "Moth Frequency Response",
    xscale = log10
)

# Define colors for the two trials
colors = [:turqoiuse, :coral]
labels = ["Before Feeding","After Feeding"]

# Get unique trials
trials = unique(grouped_data.trial)

for (i, trial) in enumerate(trials)
    trial_data = filter(:trial => ==(trial), grouped_data)
    
    # Sort data by frequency to ensure correct line plotting
    sort!(trial_data, :freq)
    
    # Plot the median line
    lines!(ax, trial_data.freq, trial_data.median, 
           color=colors[i], label=labels[i], linewidth=2)
    
    # Plot the interquartile range
    band!(ax, trial_data.freq, trial_data.q1, trial_data.q3, 
          color=(colors[i], 0.3))
end

# Add legend
axislegend(ax, "Trial", position=:lt)

# Display the plot
display(fig)

save("mungedfigs/GAINS.png",fig,px_per_unit=4)

##

phase = combine(groupby(all_moths, [:freq, :trial]), :phase => quartiles => AsTable)
##

fig = Figure(size=(800, 600), fontsize=14)
ax = Axis(fig[1, 1], 
    xlabel = "Frequency", 
    ylabel = "Phase Offset (def)",
    title = "Moth Frequency Response",
    xscale = log10
)

# Define colors for the two trials
colors = [:turqoise, :coral]
labels = ["Before Feeding","After Feeding"]

# Get unique trials
trials = unique(phase.trial)

for (i, trial) in enumerate(trials)
    trial_data = filter(:trial => ==(trial), phase)
    
    # Sort data by frequency to ensure correct line plotting
    sort!(trial_data, :freq)
    
    # Plot the median line
    lines!(ax, trial_data.freq, trial_data.median, 
           color=colors[i], label=labels[i], linewidth=2)
    
    # Plot the interquartile range
    band!(ax, trial_data.freq, trial_data.q1, trial_data.q3, 
          color=(colors[i], 0.3))
end

# Add legend
axislegend(ax, "Trial", position=:lt)

# Display the plot
display(fig)

save("mungedfigs/PHASE.png",fig,px_per_unit=4)