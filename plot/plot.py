import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Load the dataset
file_path = Path('results/deepmind-resolution-3_run-dacf296a-a138-4c07-beaa-e1a0efbab19d.csv')
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
#data.head()

# Calculate the match rate per structure for each resolution group
match_rates = data.groupby(['resolution', 'structure'])['match'].mean().reset_index()

# Pivot the table for plotting
pivot_match_rates = match_rates.pivot(index='resolution', columns='structure', values='match')

# Plotting the bar chart
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.size'] = 10
plt.figure(figsize=(12, 6))
sns.set_palette(sns.color_palette("deep", len(match_rates['structure'].unique())))
bar_plot = sns.barplot(data=match_rates, x='resolution', y='match', hue='structure', zorder=2)
for bar in bar_plot.patches:
    bar_plot.annotate(format(bar.get_height(), '.2f'), 
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                      ha='center', va='center',
                      size=9, xytext=(0, 8),
                      textcoords='offset points')
plt.title('Match Rate per Structure Grouped by Resolution')
plt.xlabel('Resolution (Pixels)')
plt.ylabel('Match Rate')
plt.legend(title='Structure')
#plt.grid(True, axis='y', which='both', linewidth=0.5, zorder=1, linestyle='-')
plt.savefig('plot/match_rate_per_structure_grouped_by_resolution.png')