import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from uuid import uuid4
import os
from matplotlib.font_manager import FontProperties

signifier_font_path = "plot/fonts/Test Signifier/TestSignifier-Medium.otf"
sohne_font_path = "plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Buch.otf"
sohne_bold_font_path = "plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Kräftig.otf"

signifier_font = FontProperties(fname=signifier_font_path)
sohne_font = FontProperties(fname=sohne_font_path)
sohne_bold_font = FontProperties(fname=sohne_bold_font_path)

# Load the dataset
file_path = Path('results/deepmind-prompts_default.csv')

def match_similarity_per_structure_grouped_by_num_nodes(file_path: Path) -> None:
    df = pd.read_csv(file_path)
    
    # Calculate overall match rate and average similarity
    overall_match_rate = df['match'].mean()
    overall_average_similarity = df['similarity'].mean()
    
    # Grouping by 'structure' and 'num_nodes' to calculate match rate and average similarity
    grouped_data = df.groupby(['structure', 'num_nodes']).agg(match_rate=('match', 'mean'), average_similarity=('similarity', 'mean')).reset_index()

    # Setting the plot size a bit larger to accommodate legends and titles better
    plt.figure(figsize=(16, 8))

    # Match Rate Visualization
    ax1 = plt.subplot(1, 2, 1)
    match_plot = sns.barplot(x='num_nodes', y='match_rate', hue='structure', data=grouped_data, palette='coolwarm', ax=ax1)
    plt.title('Accuracy of Predicted vs. Ground Truth', fontproperties=sohne_font, fontsize=12, loc='left')
    plt.xlabel('n nodes', fontproperties=sohne_font)
    plt.ylabel('accuracy', fontproperties=sohne_font)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    leg = match_plot.legend(loc='upper right', bbox_to_anchor=(1.01, 1))
    #leg.set_title('Structure', prop=sohne_font)
    for text in leg.get_texts():
        text.set_text(text.get_text().replace("_", " "))
        text.set_fontproperties(sohne_font)
    ax1.set_axisbelow(True)
    ax1.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')

    # Average Similarity Visualization
    ax2 = plt.subplot(1, 2, 2)
    similarity_plot = sns.barplot(x='num_nodes', y='average_similarity', hue='structure', data=grouped_data, palette='coolwarm', ax=ax2)
    plt.title('Similarity of Predicted vs. Ground Truth', fontproperties=sohne_font, fontsize=12, loc='left')
    plt.xlabel('n nodes', fontproperties=sohne_font)
    plt.ylabel('similarity', fontproperties=sohne_font)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    leg = similarity_plot.legend(loc='upper right', bbox_to_anchor=(1.01, 1))
    #leg.set_title('Structure', prop=sohne_font)
    for text in leg.get_texts():
        text.set_text(text.get_text().replace("_", " "))
        text.set_fontproperties(sohne_font)
    ax2.set_axisbelow(True)
    ax2.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')

    # Display overall average similarity
    plt.figtext(0.83, 0.92, f'Aggregate Mean Similarity - {overall_average_similarity:.2f}', ha='left', fontsize=10, color='red', fontproperties=sohne_font)
    # Display overall match rate
    plt.figtext(0.69, 0.92, f'Aggregate Mean Accuracy - {overall_match_rate:.2f}', ha='left', fontsize=10, color='red', fontproperties=sohne_font)

    # Enhancing the suptitle formatting
    plt.figtext(0.05, 0.92, f'{((file_path.name).replace("_", "-")).replace(".csv", "")}', va='center', fontsize=32, fontweight='bold', color='black', fontproperties=signifier_font)

    # Adjust layout for better readability
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.87])
    plt.savefig(f'plot/match_rate_and_similarity_by_structure_and_num_nodes-{(file_path.name).replace(".csv", "").upper()}.png', dpi=300)

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

def compare_match_similarity(file_path1: Path, file_path2: Path) -> None:
    # Load datasets
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    
    # Process datasets
    def process_data(df):
        overall_match_rate = df['match'].mean()
        overall_average_similarity = df['similarity'].mean()
        grouped_data = df.groupby('num_nodes').agg(match_rate=('match', 'mean'), average_similarity=('similarity', 'mean')).reset_index()
        return overall_match_rate, overall_average_similarity, grouped_data
    
    overall_match_rate1, overall_average_similarity1, grouped_data1 = process_data(df1)
    overall_match_rate2, overall_average_similarity2, grouped_data2 = process_data(df2)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Match Rate Visualization
    ax1 = plt.subplot(1, 2, 1)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    plt.plot(grouped_data1['num_nodes'], grouped_data1['match_rate'], marker='o', label='zero-shot', linestyle='-', color='#788296', lw=2.0, ms=6.0, solid_joinstyle='round')
    plt.plot(grouped_data2['num_nodes'], grouped_data2['match_rate'], marker='s', label='zero-shot-cot', linestyle='-', lw=2.0, ms=6.0, solid_joinstyle='round', color='cornflowerblue')
    plt.title('Accuracy of Predicted vs. Ground Truth', fontproperties=sohne_font, fontsize=12, loc='left')
    plt.xlabel('n nodes', fontproperties=sohne_font, fontsize=12)
    plt.ylabel('accuracy', fontproperties=sohne_font, fontsize=12)
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_text(text.get_text())
        text.set_fontproperties(sohne_font)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgrey')
    
    # Average Similarity Visualization
    ax2 = plt.subplot(1, 2, 2)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    plt.plot(grouped_data1['num_nodes'], grouped_data1['average_similarity'], marker='o', label='zero-shot', linestyle='-', color='#788296', lw=2.0, ms=6.0, solid_joinstyle='round')
    plt.plot(grouped_data2['num_nodes'], grouped_data2['average_similarity'], marker='s', label='zero-shot-cot', linestyle='-', lw=2.0, ms=6.0, solid_joinstyle='round', color='cornflowerblue')
    plt.title('Similarity of Predicted vs. Ground Truth', fontproperties=sohne_font, fontsize=12, loc='left')
    plt.xlabel('n nodes', fontproperties=sohne_font, fontsize=12)
    plt.ylabel('similarity', fontproperties=sohne_font, fontsize=12)
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_text(text.get_text())
        text.set_fontproperties(sohne_font)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgrey')
    
    plt.tight_layout()
    plt.savefig(f'plot/comparison_{file_path1.stem}_vs_{file_path2.stem}.png', dpi=300)

directory_path = Path('results')
'''
for file in directory_path.iterdir():
    if file.is_file():
        if file.name.endswith('.csv'):
            match_similarity_per_structure_grouped_by_num_nodes(file)
'''
match_similarity_per_structure_grouped_by_num_nodes(Path('results/deepmind-prompt_default.csv'))
match_similarity_per_structure_grouped_by_num_nodes(Path('results/deepmind-prompts_zero_shot_cot.csv'))
compare_match_similarity(Path('results/deepmind-prompt_default.csv'), Path('results/deepmind-prompts_zero_shot_cot.csv'))