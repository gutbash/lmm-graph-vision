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

def compare_match_similarity_by_num_nodes(file_path1: Path, file_path2: Path) -> None:
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
    

def match_similarity_by_variation_num_nodes(file_path: Path) -> None:
    
    data = pd.read_csv(file_path)

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Identify unique structures
    structures = data['structure'].unique()

    # Loop through each structure to create plots
    for structure in structures:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        for spine in axes[0].spines.values():
            spine.set_visible(False)
        for spine in axes[1].spines.values():
            spine.set_visible(False)
        fig.suptitle(f"deepmind-{structure.replace('_', '-')}", fontsize=16, fontproperties=signifier_font, x=0.175, y=0.95)

        # Filter data for the current structure
        structure_data = data[data['structure'] == structure]

        # Group by variation_id and num_nodes to calculate mean similarity and match rate
        summary = structure_data.groupby(['generation_id', 'variation_id', 'num_nodes']).agg(
            mean_similarity=('similarity', 'mean'),
            match_rate=('match', 'mean')
        ).reset_index()

        # Plot for match rate
        sns.lineplot(ax=axes[0], data=summary, x='num_nodes', y='match_rate', marker='o', lw=2.0, ms=6.0)
        axes[0].set_title('Accuracy by Number of Nodes', fontproperties=sohne_font, loc='left')
        axes[0].set_xlabel('n nodes', fontproperties=sohne_font)
        axes[0].set_ylabel('accuracy', fontproperties=sohne_font)

        # Plot for mean similarity
        sns.lineplot(ax=axes[1], data=summary, x='num_nodes', y='mean_similarity', marker='o')
        axes[1].set_title('Similarity by Number of Nodes', fontproperties=sohne_font, loc='left')
        axes[1].set_xlabel('n nodes', fontproperties=sohne_font)
        axes[1].set_ylabel('similarity', fontproperties=sohne_font)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plot/match_similarity_by_variation_num_nodes_{structure}.png')
        
def match_similarity_by_variation_generation(file_path: Path) -> None:
    
    data = pd.read_csv(file_path)

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Identify unique structures
    structures = data['structure'].unique()

    # Loop through each structure to create plots
    for structure in structures:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        for spine in axes[0].spines.values():
            spine.set_visible(False)
        for spine in axes[1].spines.values():
            spine.set_visible(False)
        fig.suptitle(f"deepmind-{structure.replace('_', '-')}", fontsize=16, fontproperties=signifier_font, x=0.175, y=0.95)

        # Filter data for the current structure
        structure_data = data[data['structure'] == structure]

        # Group by variation_id and num_nodes to calculate mean similarity and match rate
        summary = structure_data.groupby(['generation_id', 'variation_id']).agg(
            mean_similarity=('similarity', 'mean'),
            match_rate=('match', 'mean')
        ).reset_index()

        # Plot for match rate
        sns.lineplot(ax=axes[0], data=summary, x='generation_id', y='match_rate', marker='o', lw=2.0, ms=6.0)
        axes[0].set_title('Accuracy by Generation with Variation Deviations', fontproperties=sohne_font, loc='left')
        axes[0].set_xlabel('generation', fontproperties=sohne_font)
        axes[0].set_ylabel('accuracy', fontproperties=sohne_font)

        # Plot for mean similarity
        sns.lineplot(ax=axes[1], data=summary, x='generation_id', y='mean_similarity', marker='o')
        axes[1].set_title('Similarity by Generation with Variation Deviations', fontproperties=sohne_font, loc='left')
        axes[1].set_xlabel('generation', fontproperties=sohne_font)
        axes[1].set_ylabel('similarity', fontproperties=sohne_font)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plot/match_similarity_by_variation_num_nodes_{structure}.png')
        
def match_similarity_by_arrow(file_path1: Path, file_path2: Path) -> None:
        
    df_openai = pd.read_csv(file_path1)
    df_deepmind = pd.read_csv(file_path2)

    def prepare_data(df):
        # Calculate mean of `match` and `similarity` grouped by `arrow_style`
        # For `match`, first convert boolean to int
        df['match'] = df['match'].astype(int)
        grouped = df.groupby('arrow_style').agg({'match': 'mean', 'similarity': 'mean'}).reset_index()
        return grouped

    # Prepare data
    data_openai = prepare_data(df_openai)
    data_deepmind = prepare_data(df_deepmind)
    
    # Adjusting the data scale for similarity
    data_openai['similarity'] = data_openai['similarity'] / 100
    data_deepmind['similarity'] = data_deepmind['similarity'] / 100

    # Plot settings
    sns.set_theme(style="whitegrid", palette="coolwarm")

    # Recreate figure for the adjusted axis orientation
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=300)

    # Adjusted Dot plots with switched axes
    sns.stripplot(y="match", x="arrow_style", data=data_openai, ax=axes[0, 0], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="match", x="arrow_style", data=data_deepmind, ax=axes[0, 1], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="similarity", x="arrow_style", data=data_openai, ax=axes[1, 0], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="similarity", x="arrow_style", data=data_deepmind, ax=axes[1, 1], size=10, jitter=True, palette="coolwarm", marker='o')

    for ax in axes.flat:
        
        # Set font for all tick labels
        for label in ax.get_yticklabels():
            label.set_fontproperties(sohne_font)
    
    # Titles and customization after switching axes
    #fig.suptitle('directed graph by arrow style', fontsize=16, fontproperties=signifier_font)
    axes[0, 0].set_title('OpenAI', fontsize=24, fontproperties=sohne_font)
    axes[0, 1].set_title('DeepMind', fontsize=24, fontproperties=sohne_font)
    axes[0, 0].set_ylim(0, 1)
    #axes[0, 1].set_title('OpenAI - Similarity', fontsize=14)
    axes[0, 1].set_ylim(0, 1)
    #axes[1, 0].set_title('DeepMind - Match Rate', fontsize=14)
    axes[1, 0].set_ylim(0, 1)
    #axes[1, 1].set_title('DeepMind - Similarity', fontsize=14)
    axes[1, 1].set_ylim(0, 1)
    
    axes[0, 0].set_ylabel('Accuracy', fontproperties=sohne_font)
    axes[1, 0].set_ylabel('Similarity', fontproperties=sohne_font)
    #axes[0, 1].set_ylabel('Accuracy', fontproperties=sohne_font)
    #axes[1, 1].set_ylabel('Similarity', fontproperties=sohne_font)
    
    for ax in axes.flat:
        #ax.set_xlabel('Arrow Style')
        ax.label_outer()

    # Adjusting layout and axes for clarity with switched orientation
    plt.tight_layout(rect=[0, 0, 1, 1])
    for ax in axes.flat:
        ax.set_xlabel('')
        
    plt.savefig(f'plot/match_similarity_by_arrow_style.png', dpi=300)

match_similarity_by_arrow(Path('results/openai-prompts_zero_shot-arrows.csv'), Path('results/deepmind-prompts_zero_shot-arrows.csv'))