import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from uuid import uuid4

# Load the dataset
file_path = Path('results/deepmind-9b121cc7-045c-4e33-9351-363d5febbea0.csv')

def match_rate_per_structure_grouped_by_resolution(file_path: Path) -> None:
    """
    Plots the match rate per structure grouped by resolution.
    
    Parameters
    ----------
    file_path : Path
        the path to the results CSV file
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display the first few rows of the dataframe to understand its structure
    #data.head()

    # Calculate the match rate per structure for each resolution group
    match_rates = data.groupby(['resolution', 'structure'])['similarity'].mean().reset_index()
    
    print(match_rates[(match_rates['resolution'] == 256) & (match_rates['structure'] == 'undirected_graph')])

    # Pivot the table for plotting
    pivot_match_rates = match_rates.pivot(index='resolution', columns='structure', values='similarity')
    print(pivot_match_rates)

    # Plotting the bar chart
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
    plt.rcParams['font.size'] = 10
    plt.figure(figsize=(12, 6))
    sns.set_palette(sns.color_palette("deep", len(match_rates['structure'].unique())))
    bar_plot = sns.barplot(data=match_rates, x='resolution', y='similarity', hue='structure', zorder=2)
    for bar in bar_plot.patches:
        if bar.get_height() > 0:
            bar_plot.annotate(format(bar.get_height(), '.2f'), 
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                            ha='center', va='center',
                            size=9, xytext=(0, 8),
                            textcoords='offset points')
    plt.title('Average Similarity Rate per Structure Grouped by Resolution')
    plt.xlabel('Resolution (Pixels)')
    plt.ylabel('Similarity Rate')
    plt.legend(title='Structure')
    #plt.grid(True, axis='y', which='both', linewidth=0.5, zorder=1, linestyle='-')
    plt.savefig(f'plot/similarity_rate_per_structure_grouped_by_resolution-{(file_path.name).replace(".csv", "").capitalize()}.png')
    plt.close()

def match_rate_per_num_nodes_and_resolution(file_path: Path) -> None:
    
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Calculate the probability of match for each combination of num_nodes and resolution
    prob_df = data.groupby(['num_nodes', 'resolution'])['similarity'].mean().reset_index()

    # Pivot the dataframe to create a matrix suitable for heatmap plotting
    pivot_df = prob_df.pivot(index="num_nodes", columns="resolution", values="similarity")

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Average Similarity Rate by Number of Nodes and Resolution')
    plt.xlabel('Resolution')
    plt.ylabel('Number of Nodes')
    #plt.show()
    plt.savefig(f'plot/probability_of_similarity_by_num_nodes_and_resolution-{(file_path.name).replace(".csv", "").capitalize()}.png')
    plt.close()
    
def similarity_heatmap(file_path: Path) -> None:
    
    df = pd.read_csv(file_path)
    
    print()
    
    # Group by structure and num_nodes, then calculate the mean similarity
    grouped_df = df.groupby(['structure', 'num_nodes'])['similarity'].mean().unstack().fillna(0)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(grouped_df, annot=True, fmt=".1f", cmap='magma', linewidths=.5)
    plt.title('Average Similarity Rate by Structure and Number of Nodes', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Structure', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(f'plot/heatmap_of_average_similarity_by_structure_and_num_nodes-{(file_path.name).replace(".csv", "").capitalize()}.png')
    
match_rate_per_structure_grouped_by_resolution(file_path)
match_rate_per_num_nodes_and_resolution(file_path)
similarity_heatmap(file_path)