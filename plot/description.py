from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from uuid import uuid4
import os
from matplotlib.font_manager import FontProperties, fontManager

signifier_font_path = Path("plot/fonts/Test Signifier/TestSignifier-Medium.otf")
sohne_font_path = Path("plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Buch.otf")
sohne_bold_font_path = Path("plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Kräftig.otf")
computer_modern_font_path = Path("plot/fonts/Computer Modern/cmunbx.ttf")

signifier_font = FontProperties(fname=signifier_font_path)
sohne_font = FontProperties(fname=sohne_font_path)
sohne_bold_font = FontProperties(fname=sohne_bold_font_path)
computer_modern_font = FontProperties(fname=computer_modern_font_path)

signifier_font_name = signifier_font.get_name()
sohne_font_name = sohne_font.get_name()
sohne_bold_font_name = sohne_bold_font.get_name()
computer_modern_font_name = computer_modern_font.get_name()

# Register the fonts with Matplotlib's font manager
fontManager.addfont(signifier_font_path)
fontManager.addfont(sohne_font_path)
fontManager.addfont(sohne_bold_font_path)
fontManager.addfont(computer_modern_font_path)

plt.rcParams['font.family'] = sohne_font_name

# Load the dataset
file_path = Path('results/deepmind-prompts_default.csv')

def accuracy_by_num_nodes(file_path_1: Path, file_path_2: Path) -> None:
    # Read each CSV file into a DataFrame
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # Prepare a figure to contain subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Now sharing the y-axis

    models = ['gpt-4-vision-preview', 'gemini-pro-vision']
    
    axes[0].set_ylabel('Accuracy')
    
    model_colors = [(230/255,110/255,180/255,0.15), (179/255,110/255,230/255,0.15)]

    for subplot_index, (df, ax) in enumerate(zip([df1, df2], axes), start=1):
        # Group by 'structure' and 'num_nodes' to calculate match rate
        grouped_data = df.groupby(['structure', 'num_nodes']).agg(match_rate=('match', 'mean')).reset_index()
        # normalize match to 0-100 scale
        grouped_data['match_rate'] = grouped_data['match_rate'] * 100

        # Get unique structures and num_nodes
        structures = grouped_data['structure'].unique()
        num_nodes = grouped_data['num_nodes'].unique()

        # Set x-tick positions
        x_ticks = np.arange(len(num_nodes))

        # Define a custom color palette
        color_palette = sns.color_palette('coolwarm', n_colors=len(grouped_data.columns))
        color_palette = ['#22577a', '#38a3a5', '#57cc99', '#80ed99']
        color_palette = ['#2fff00', '#00e0e9', '#00aaff', '#9500ff']
        color_palette = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

        # Calculate the width of each bar
        bar_width = 0.8 / len(structures)

        # Plot bars for each structure
        for i, structure in enumerate(structures):
            structure_data = grouped_data[grouped_data['structure'] == structure]

            # Calculate the positions for the bars of each structure
            bar_positions = x_ticks + (i - len(structures) / 2 + 0.5) * bar_width

            ax.bar(bar_positions, structure_data['match_rate'], width=bar_width, alpha=1,
                   color=color_palette[i % len(color_palette)],
                   label=structure.replace("_", " ") if subplot_index == 2 else "")

        # Set x-tick labels and positions
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(num_nodes)

        ax.set_xlabel('Number of Nodes')

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_axisbelow(True)
        ax.grid(True, which='both', axis='y', linestyle='-', linewidth=1, color='lightgrey', alpha=0.25)

        ax.set_ylim(0, 100)
        
        model_name = f'{models[subplot_index - 1]}'

        # Setting the title for each subplot
        ax.set_title(model_name, fontsize=10, loc='left', pad=15, bbox=dict(facecolor=model_colors[subplot_index-1], edgecolor='none', boxstyle='round,pad=0.3,rounding_size=0.7'))

    axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1.25))
    plt.suptitle('Zero-shot accuracy by number of nodes', fontproperties=sohne_bold_font, fontsize=16, x=0.22, y=0.85)

    # Adjust layout for better readability
    plt.tight_layout(rect=[0.00, 0.00, 1, 1])

    # Save the figure
    plt.savefig('plot/accuracy_by_num_nodes.pdf', dpi=300, transparent=True, bbox_inches='tight', format='pdf')
    
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
    plt.title('Accuracy of Predicted vs. Ground Truth', fontsize=12, loc='left')
    plt.xlabel('n nodes', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_text(text.get_text())
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgrey')
    
    # Average Similarity Visualization
    ax2 = plt.subplot(1, 2, 2)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    plt.plot(grouped_data1['num_nodes'], grouped_data1['average_similarity'], marker='o', label='zero-shot', linestyle='-', color='#788296', lw=2.0, ms=6.0, solid_joinstyle='round')
    plt.plot(grouped_data2['num_nodes'], grouped_data2['average_similarity'], marker='s', label='zero-shot-cot', linestyle='-', lw=2.0, ms=6.0, solid_joinstyle='round', color='cornflowerblue')
    plt.title('Similarity of Predicted vs. Ground Truth', fontsize=12, loc='left')
    plt.xlabel('n nodes', fontsize=12)
    plt.ylabel('similarity', fontsize=12)
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_text(text.get_text())
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgrey')
    
    plt.tight_layout()
    plt.savefig(f'plot/comparison_{file_path1.stem}_vs_{file_path2.stem}.png', dpi=300, transparent=True)
    

def match_similarity_by_variation_num_nodes(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    
    data['similarity'] = data['similarity'] / 100
    
    structures = data['structure'].unique()
    num_structures = len(structures)
    
    fig, axes = plt.subplots(1, num_structures, figsize=(10, 3), dpi=300, sharex=True, sharey=True)
    
    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]
        
        summary = structure_data.groupby(['generation_id', 'variation_id', 'num_nodes']).agg(
            match_rate=('match', 'mean')
        ).reset_index()
        
        sns.lineplot(ax=axes[i], data=summary, x='num_nodes', y='match_rate', marker='o', lw=2.0, ms=6.0)
        axes[i].set_title(f"{structure.replace('_', ' ')}", loc='left')
        axes[i].set_xlabel('Number of Nodes')
        axes[i].set_ylabel('Accuracy')
        
        # Hide top and right spines for each subplot
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['left'].set_color((0, 0, 0, 0.2))
        axes[i].spines['bottom'].set_color((0, 0, 0, 0.2))
        
        axes[i].set_xticks([3, 4, 5, 6, 7, 8, 9])
    
    plt.title('Mean Predicted vs Ground Truth Performance by Variation per Number of Nodes with Standard Deviation', fontsize=12, x=0.04, y=0.88)
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('plot/match_by_variation_num_nodes.png')
        
def match_similarity_by_variation_generation(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    #sns.set_style("whitegrid")
    #sns.set_theme(style="whitegrid")

    data['similarity'] = data['similarity'] / 100
    structures = data['structure'].unique()
    num_structures = len(structures)

    fig, axes = plt.subplots(2, num_structures, figsize=(12, 6), dpi=300, sharex=True, sharey=True)
    #fig.suptitle("Accuracy and Similarity by Generation with Variation Deviations", fontsize=16, y=0.95)

    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]
        summary = structure_data.groupby(['generation_id', 'variation_id']).agg(
            mean_similarity=('similarity', 'mean'),
            match_rate=('match', 'mean')
        ).reset_index()

        sns.lineplot(ax=axes[0, i], data=summary, x='generation_id', y='match_rate', marker='o', lw=2.0, ms=6.0)
        axes[0, i].set_title(f"{structure.replace('_', ' ')}", loc='left')
        axes[0, i].set_xlabel('generation')
        axes[0, i].set_ylabel('accuracy')

        sns.lineplot(ax=axes[1, i], data=summary, x='generation_id', y='mean_similarity', marker='o')
        axes[1, i].set_xlabel('generation')
        axes[1, i].set_ylabel('similarity')
        
        for ax in axes.flat:
            # Hide top and right spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color((0, 0, 0, 0.2))
            ax.spines['bottom'].set_color((0, 0, 0, 0.2))

    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Variation per Generation with Standard Deviation', fontsize=12, x=0.05, y=0.90, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('plot/match_similarity_by_variation_generation.png')
        
def match_similarity_by_arrow(file_path1: Path, file_path2: Path) -> None:
        
    df_1 = pd.read_csv(file_path1)
    df_2 = pd.read_csv(file_path2)

    def prepare_data(df):
        # Calculate mean of `match` and `similarity` grouped by `arrow_style`
        # For `match`, first convert boolean to int
        df['match'] = df['match'].astype(int)
        grouped = df.groupby('arrow_style').agg({'match': 'mean', 'similarity': 'mean'}).reset_index()
        return grouped

    # Prepare data
    data_1 = prepare_data(df_1)
    data_2 = prepare_data(df_2)
    
    # Adjusting the data scale for similarity
    data_1['similarity'] = data_1['similarity'] / 100
    data_2['similarity'] = data_2['similarity'] / 100

    # Plot settings
    sns.set_theme(style="whitegrid", palette="coolwarm")

    # Recreate figure for the adjusted axis orientation
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=300)

    # Adjusted Dot plots with switched axes
    sns.stripplot(y="match", x="arrow_style", data=data_1, ax=axes[0, 0], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="match", x="arrow_style", data=data_2, ax=axes[0, 1], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="similarity", x="arrow_style", data=data_1, ax=axes[1, 0], size=10, jitter=True, palette="coolwarm", marker='o')
    sns.stripplot(y="similarity", x="arrow_style", data=data_2, ax=axes[1, 1], size=10, jitter=True, palette="coolwarm", marker='o')
    
    # Titles and customization after switching axes
    #fig.suptitle('directed graph by arrow style', fontsize=16)
    axes[0, 0].set_title(f'{file_path1.stem}', fontsize=24)
    axes[0, 1].set_title(f'{file_path2.stem}', fontsize=24)
    axes[0, 0].set_ylim(0, 1)
    #axes[0, 1].set_title('OpenAI - Similarity', fontsize=14)
    axes[0, 1].set_ylim(0, 1)
    #axes[1, 0].set_title('DeepMind - Match Rate', fontsize=14)
    axes[1, 0].set_ylim(0, 1)
    #axes[1, 1].set_title('DeepMind - Similarity', fontsize=14)
    axes[1, 1].set_ylim(0, 1)
    
    axes[0, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylabel('Similarity')
    #axes[0, 1].set_ylabel('Accuracy')
    #axes[1, 1].set_ylabel('Similarity')
    
    for ax in axes.flat:
        #ax.set_xlabel('Arrow Style')
        ax.label_outer()

    # Adjusting layout and axes for clarity with switched orientation
    plt.tight_layout(rect=[0, 0, 1, 1])
    for ax in axes.flat:
        ax.set_xlabel('')
        
    plt.savefig(f'plot/match_similarity_by_arrow_style.png', dpi=300, transparent=True)
    
def match_similarity_by_color(file_path: Path) -> None:
        
    df = pd.read_csv(file_path)

    def prepare_data(df):
        # Calculate mean of `match` and `similarity` grouped by `arrow_style`
        # For `match`, first convert boolean to int
        df['match'] = df['match'].astype(int)
        grouped = df.groupby('node_color').agg({'match': 'mean', 'similarity': 'mean'}).reset_index()
        return grouped

    # Prepare data
    data = prepare_data(df)
    
    # Adjusting the data scale for similarity
    data['similarity'] = data['similarity'] / 100

    # Plot settings
    sns.set_theme(style="whitegrid", palette="coolwarm")

    # Recreate figure for the adjusted axis orientation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=300)

    # Adjusted Dot plots with switched axes
    sns.barplot(y="match", x="node_color", data=data, ax=axes[0], palette="coolwarm")
    sns.barplot(y="similarity", x="node_color", data=data, ax=axes[1], palette="coolwarm")

    for ax in axes.flat:
        
        for label in ax.get_xticklabels():
            label.set_text((label.get_text()).replace('#', ''))
    
    # Titles and customization after switching axes
    fig.suptitle('deepmind-zero-shot-color', fontsize=16, y=0.95)
    #axes[0, 0].set_title('OpenAI', fontsize=24)
    #axes[0, 1].set_title('DeepMind', fontsize=24)
    axes[0].set_ylim(0, 0.8)
    axes[1].set_ylim(0, 0.8)
    
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Similarity')
    axes[0].set_xlabel('Node Color')
    axes[1].set_xlabel('Node Color')
    
    for ax in axes.flat:
        # Hide top and right spines for each subplot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color((0, 0, 0, 0.2))
        ax.spines['bottom'].set_color((0, 0, 0, 0.2))

    # Adjusting layout and axes for clarity with switched orientation
    plt.tight_layout(rect=[0, 0, 1, 1])
        
    plt.savefig(f'plot/match_similarity_by_color.png', dpi=300, transparent=True)
    
def match_similarity_by_width(file_path: Path) -> None:
        
    df = pd.read_csv(file_path)

    def prepare_data(df):
        # Calculate mean of `match` and `similarity` grouped by `arrow_style`
        # For `match`, first convert boolean to int
        df['match'] = df['match'].astype(int)
        grouped = df.groupby('edge_width').agg({'match': 'mean', 'similarity': 'mean'}).reset_index()
        return grouped

    # Prepare data
    data = prepare_data(df)
    
    # Adjusting the data scale for similarity
    data['similarity'] = data['similarity'] / 100

    # Plot settings
    sns.set_theme(style="whitegrid", palette="coolwarm")

    # Recreate figure for the adjusted axis orientation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=300)

    # Adjusted Dot plots with switched axes
    sns.barplot(y="match", x="edge_width", data=data, ax=axes[0], palette="coolwarm")
    sns.barplot(y="similarity", x="edge_width", data=data, ax=axes[1], palette="coolwarm")

    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_text((label.get_text()).replace('#', ''))
    
    # Titles and customization after switching axes
    fig.suptitle((file_path.stem).replace('_', '-'), fontsize=16, y=0.95)
    #axes[0, 0].set_title('OpenAI', fontsize=24)
    #axes[0, 1].set_title('DeepMind', fontsize=24)
    axes[0].set_ylim(0, 0.8)
    axes[1].set_ylim(0, 0.8)
    
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Similarity')
    axes[0].set_xlabel('Edge Width')
    axes[1].set_xlabel('Edge Width')
    
    for ax in axes.flat:
        # Hide top and right spines for each subplot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color((0, 0, 0, 0.2))
        ax.spines['bottom'].set_color((0, 0, 0, 0.2))

    # Adjusting layout and axes for clarity with switched orientation
    plt.tight_layout(rect=[0, 0, 1, 1])
        
    plt.savefig(f'plot/match_similarity_by_width.png', dpi=300, transparent=True)
    
def heatmap_match_similarity_by_width_and_color(file_path: Path) -> None:
    data = pd.read_csv(file_path)

    data['node_color'] = data['node_color'].str.replace('#ff0000', 'red').str.replace('#ffff00', 'yellow').str.replace('#ffffff', 'white')
    data['similarity'] = data['similarity'] / 100

    structures = data['structure'].unique()

    figs, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300, sharex=True, sharey=True)

    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]

        # Heatmap of match and similarity grouped by structure of edge_width by node_color
        # Preparing the data
        heatmap_data_match = structure_data.groupby(['edge_width', 'node_color'])['similarity'].mean().unstack().fillna(0)
        heatmap_data_similarity = structure_data.groupby(['edge_width', 'node_color'])['match'].mean().unstack().fillna(0)

        # Heatmap for match
        sns.heatmap(data=heatmap_data_match, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0, ax=axes[0, i], vmin=0, vmax=1)
        axes[0, i].set_title(f'similarity of {(structure).replace("_", " ")}', fontsize=12)
        axes[0, i].set_xlabel('', fontsize=10)
        axes[0, i].set_ylabel('', fontsize=10)

        # Heatmap for similarity
        sns.heatmap(data=heatmap_data_similarity, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0, ax=axes[1, i], vmin=0, vmax=1)
        axes[1, i].set_title(f'accuracy of {(structure).replace("_", " ")}', fontsize=12)
        axes[1, i].set_xlabel('node color', fontsize=10)
        axes[1, i].set_ylabel('', fontsize=10)
        
    axes[0, 0].set_ylabel('edge width', fontsize=10)
    axes[1, 0].set_ylabel('edge width', fontsize=10)

    for ax in axes.flat:
        # remove '#' from the x tick labels
        for label in ax.get_xticklabels():
            label.set_text(label.get_text().replace('#', ''))

    figs.text(0.04, 0.93, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    figs.suptitle('Mean Predicted vs Ground Truth Performance by Edge Width and Node Color', fontsize=12, x=0.04, y=0.91, ha='left')

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    plt.savefig(f'plot/heatmap_of_match_and_similarity_by_width_and_color.png', dpi=300, transparent=True)
    
def line_plot_match_similarity_by_edge_width(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    data['similarity'] = data['similarity'] / 100
    data['structure'] = data['structure'].replace('_', ' ', regex=True)
    
    structures = data['structure'].unique()
    
    fig, axes = plt.subplots(2, len(structures), figsize=(12, 6), sharex=True, sharey=True)
    
    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]
        
        sns.lineplot(data=structure_data, x='edge_width', y='match', estimator='mean', ci='sd', ax=axes[0, i], marker='o')
        axes[0, i].set_ylabel('accuracy')
        axes[0, i].set_xlabel('edge width')
        axes[0, i].set_title(structure, loc='left')
        
        sns.lineplot(data=structure_data, x='edge_width', y='similarity', estimator='mean', ci='sd', ax=axes[1, i], marker='o')
        axes[1, i].set_ylabel('similarity')
        axes[1, i].set_xlabel('edge width')
        #axes[i, 1].set_title(structure)
        
        axes[0, i].set_ylim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[0, i].set_xticks([1.0, 3.0, 5.0])
        axes[1, i].set_xticks([1.0, 3.0, 5.0])
        
        for ax in axes.flat:
            # Hide top and right spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color((0, 0, 0, 0.2))
            ax.spines['bottom'].set_color((0, 0, 0, 0.2))
    
    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Edge Width with Standard Deviation', fontsize=12, x=0.05, y=0.90, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f'plot/match_similarity_by_edge_width.png', dpi=300, transparent=True)


def line_plot_accuracy_by_num_nodes(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    data['similarity'] = data['similarity'] / 100
    data['structure'] = data['structure'].replace('_', ' ', regex=True)
    
    structures = data['structure'].unique()
    
    fig, axes = plt.subplots(2, len(structures), figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle('Match and Similarity by Number of Nodes', fontsize=16)
    
    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]
        
        sns.lineplot(data=structure_data, x='num_nodes', y='match', estimator='mean', ci='sd', ax=axes[0, i], marker='o')
        axes[0, i].set_ylabel('accuracy')
        axes[0, i].set_xlabel('number of nodes')
        axes[0, i].set_title(structure, loc='left')
        
        sns.lineplot(data=structure_data, x='num_nodes', y='similarity', estimator='mean', ci='sd', ax=axes[1, i], marker='o')
        axes[1, i].set_ylabel('similarity')
        axes[1, i].set_xlabel('number of nodes')
        #axes[i, 1].set_title(structure)
        
        axes[0, i].set_ylim(0, 1)
        axes[1, i].set_ylim(0, 1)
        
        axes[0, i].set_xticks([3, 6, 9])
        axes[1, i].set_xticks([3, 6, 9])
        
        for ax in axes.flat:
            # Hide top and right spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color((0, 0, 0, 0.2))
            ax.spines['bottom'].set_color((0, 0, 0, 0.2))
            
        axes[0, i].set_xticks([3, 4, 5, 6, 7, 8, 9])
        axes[1, i].set_xticks([3, 4, 5, 6, 7, 8, 9])
    
    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Number of Nodes with Standard Deviation', fontsize=12, x=0.05, y=0.90, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f'plot/match_similarity_by_num_nodes.png', dpi=300, transparent=True)


def bar_plot_match_similarity_by_color(file_path: Path) -> None:
    # Bar plot of match and similarity grouped by structure of node_color
    data = pd.read_csv(file_path)
    data['similarity'] = data['similarity'] / 100
    data['structure'] = data['structure'].replace('_', ' ', regex=True)
    data['node_color'] = data['node_color'].str.replace('#ff0000', 'red').str.replace('#ffff00', 'yellow').str.replace('#ffffff', 'white')

    structures = data['structure'].unique()
    num_structures = len(structures)

    fig, axes = plt.subplots(2, num_structures, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle('Accuracy and Similarity by Node Color', fontsize=16, y=0.95)

    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]

        sns.barplot(data=structure_data, x='node_color', y='match', estimator=np.mean, ci='sd', ax=axes[0, i], palette='viridis')

        axes[0, i].set_title(f'{structure}', fontsize=12, pad=10, fontproperties=sohne_bold_font)

        sns.barplot(data=structure_data, x='node_color', y='similarity', estimator=np.mean, ci='sd', ax=axes[1, i], palette='viridis')
        
        axes[0, i].set_xlabel('node color')
        axes[1, i].set_xlabel('node color')
        axes[0, i].set_ylabel('accuracy')
        axes[1, i].set_ylabel('similarity')

        #axes[1, i].set_title(f'{structure}', fontsize=12, pad=10, fontproperties=sohne_bold_font)

        axes[0, i].set_ylim(0, 1)
        axes[1, i].set_ylim(0, 1)

    for ax in axes.flat:
        # Hide top and right spines for each subplot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color((0, 0, 0, 0.2))
        ax.spines['bottom'].set_color((0, 0, 0, 0.2))

    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Node Color', fontsize=12, x=0.05, y=0.90, ha='left')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f'plot/match_similarity_by_color.png', dpi=300, transparent=True)

def bar_plot_accuracy_by_task_stacked(file_path1: Path, file_path2: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    
    #adjust subplot spacing
    plt.subplots_adjust(wspace=-1.1)
    
    models = ['gpt-4-vision-preview', 'gemini-pro-vision']

    for idx, file_path in enumerate([file_path1, file_path2]):
        data = pd.read_csv(file_path)
        data['similarity'] = data['similarity'] / 100
        data['structure'] = data['structure'].replace('_', ' ', regex=True)
        data['task'] = data['task'].str.replace('_', ' ', regex=True)

        # Calculate the mean of 'match' for each task and structure combination
        match_rate = data.groupby(['task', 'structure'])['match'].mean().unstack()

        # Generate a color palette from the following list
        color_palette = ['#0080FF', '#FF00FF', '#FF7F00', '#00FF80']
        palette = sns.color_palette(color_palette, n_colors=len(match_rate.columns))

        # The width of the bars
        bar_width = 0.7

        for task in match_rate.index:
            # Sort structures by their match rate for the current task in descending order
            sorted_structures = match_rate.loc[task].sort_values(ascending=False).index

            for structure in sorted_structures:
                value = match_rate.loc[task, structure]
                # Get the color for the current structure
                color = palette[match_rate.columns.get_loc(structure)]
                # Plot the bar for the current structure
                axs[idx].bar(task, value, color=color, label=structure, width=bar_width, alpha=1)
        
        axs[idx].set_axisbelow(True)
        for spine in ['top', 'right', 'left', 'bottom']:
            axs[idx].spines[spine].set_visible(False)
        #axs[idx].spines['bottom'].set_visible(True)
        
        axs[idx].grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')

        axs[idx].set_title(f'{models[idx]}', fontsize=12, loc='left', pad=12.5)
        axs[idx].set_ylim(0, 1)
        axs[idx].set_xticks(range(len(match_rate.index)))
        axs[idx].set_xticklabels(match_rate.index, rotation=45, ha="right")
        axs[idx].set_ylabel('Accuracy' if idx == 0 else '')
        # Ensure the legend is only added once per subplot
        if idx == 1:
            # manually create the legend with square color patches
            legend_elements = [Patch(facecolor=palette[i], label=structure) for i, structure in enumerate(match_rate.columns)]
            axs[idx].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.275))
    
    plt.suptitle('Zero-shot accuracy by task', fontproperties=sohne_bold_font, fontsize=16, x=0.081, y=0.87, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 1.05])
    plt.savefig(f'plot/match_similarity_by_task_overlay_sorted.pdf', dpi=300, transparent=True, format='pdf', bbox_inches='tight')
    
def accuracy_by_task(file_path1: Path, file_path2: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    
    #adjust subplot spacing
    plt.subplots_adjust(wspace=-1.1)
    
    models = ['gpt-4-vision-preview', 'gemini-pro-vision']
    
    model_colors = [(230/255,110/255,180/255,0.15), (179/255,110/255,230/255,0.15)]

    for idx, file_path in enumerate([file_path1, file_path2]):
        data = pd.read_csv(file_path)
        data['similarity'] = data['similarity'] / 100
        data['structure'] = data['structure'].replace('_', ' ', regex=True)
        data['task'] = data['task'].str.replace('_', ' ', regex=True)

        # Calculate the mean of 'match' for each task and structure combination
        match_rate = data.groupby(['task', 'structure'])['match'].mean().unstack()
        # normalize to 0-100 scale
        match_rate = match_rate * 100
        # switch the two halfs of the dataframe
        match_rate = pd.concat([match_rate.iloc[:, 2:], match_rate.iloc[:, :2]], axis=1)

        # Generate a color palette from the following list
        color_palette = ['#22577a', '#38a3a5', '#57cc99', '#80ed99']
        color_palette = ['#2fff00', '#00e0e9', '#00aaff', '#9500ff']
        color_palette = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
        palette = sns.color_palette(color_palette, n_colors=len(match_rate.columns))

        # The width of the bars
        bar_width = 1.5 / len(match_rate.columns)
        
        for j, task in enumerate(match_rate.index):
            non_zero_structures = match_rate.loc[task][match_rate.loc[task] > 0].index
            num_non_zero = len(non_zero_structures)
            
            for i, structure in enumerate(non_zero_structures):
                value = match_rate.loc[task, structure]
                # Get the color for the current structure
                color = palette[match_rate.columns.get_loc(structure)]
                # Calculate the position of the bar
                pos = j - (num_non_zero - 1) * bar_width / 2 + i * bar_width
                # Plot the bar for the current structure
                axs[idx].bar(pos, value, color=color, label=structure, width=bar_width, alpha=1)
        
        axs[idx].set_axisbelow(True)
        for spine in ['top', 'right', 'left', 'bottom']:
            axs[idx].spines[spine].set_visible(False)
        #axs[idx].spines['bottom'].set_visible(True)
        
        axs[idx].grid(True, which='both', axis='y', linestyle='-', linewidth=1, color='lightgrey', alpha=0.25)

        axs[idx].set_title(f'{models[idx]}', fontsize=10, loc='left', pad=12.5, bbox=dict(facecolor=model_colors[idx], edgecolor='none', boxstyle='round,pad=0.3,rounding_size=0.7'))
        axs[idx].set_ylim(0, 100)
        axs[idx].set_xticks(range(len(match_rate.index)))
        axs[idx].set_xticklabels(match_rate.index, rotation=45, ha="right")
        axs[idx].set_ylabel('Accuracy' if idx == 0 else '')
        # Ensure the legend is only added once per subplot
        if idx == 1:
            # manually create the legend with square color patches
            legend_elements = [Patch(facecolor=palette[i], label=structure) for i, structure in enumerate(match_rate.columns)]
            axs[idx].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.275))
    
    plt.suptitle('Zero-shot accuracy by task', fontproperties=sohne_bold_font, fontsize=16, x=0.081, y=0.87, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 1.05])
    plt.savefig(f'plot/accuracy_by_task.pdf', dpi=300, transparent=True, format='pdf', bbox_inches='tight')
    
def calculate_accuracies(file_path1, file_path2):
    # Load the datasets
    data1 = pd.read_csv(file_path1)
    data2 = pd.read_csv(file_path2)
    
    # Add a column to each dataframe to identify the model (source file)
    data1['model'] = file_path1.stem.split('-')[0]
    data2['model'] = file_path2.stem.split('-')[0]
    
    # Combine both datasets for easier manipulation
    combined_data = pd.concat([data1, data2])
    
    # Helper function to calculate accuracy based on match criteria
    def accuracy_for_criteria(group, attempts):
        # For pass@1, check if there's a match in the first attempt
        if attempts == 1:
            return group.sort_values('attempt_id').head(1)['match'].mean()
        
        # For pass@3, check if there's a match within the first 3 attempts
        elif attempts == 3:
            return group.sort_values('attempt_id').head(3)['match'].any().astype(int)
    
    # Calculating accuracy
    def calculate_accuracy(df, attempts):
        results = df.groupby(['structure', 'model', 'format_id', 'variation_id', 'generation_id', 'task']).apply(accuracy_for_criteria, attempts=attempts)
        return results.groupby(['structure', 'model']).mean(), results.groupby('model').mean()

    # Calculating accuracies for Pass@1 and Pass@3
    accuracy_pass1_structure_model, accuracy_pass1_overall_model = calculate_accuracy(combined_data, 1)
    accuracy_pass3_structure_model, accuracy_pass3_overall_model = calculate_accuracy(combined_data, 3)
    
    # Formatting results
    accuracy_pass1_structure_model = accuracy_pass1_structure_model.reset_index().rename(columns={0: 'Pass@1 Accuracy'})
    accuracy_pass3_structure_model = accuracy_pass3_structure_model.reset_index().rename(columns={0: 'Pass@3 Accuracy'})
    accuracy_results_structure_model = pd.merge(accuracy_pass1_structure_model, accuracy_pass3_structure_model, on=['structure', 'model'])
    
    accuracy_pass1_overall_model = accuracy_pass1_overall_model.reset_index().rename(columns={0: 'Pass@1 Overall Accuracy'})
    accuracy_pass3_overall_model = accuracy_pass3_overall_model.reset_index().rename(columns={0: 'Pass@3 Overall Accuracy'})
    accuracy_results_overall_model = pd.merge(accuracy_pass1_overall_model, accuracy_pass3_overall_model, on='model')
    
    # Calculating accuracies for grouped structures
    def calculate_grouped_accuracy(df, attempts, group_mapping):
        df['grouped_structure'] = df['structure'].map(group_mapping)
        results = df.groupby(['grouped_structure', 'model', 'format_id', 'variation_id', 'generation_id', 'task']).apply(accuracy_for_criteria, attempts=attempts)
        return results.groupby(['grouped_structure', 'model']).mean()

    group_mapping = {
        'binary_tree': 'binary_tree_group',
        'binary_search_tree': 'binary_tree_group',
        'directed_graph': 'graph_group',
        'undirected_graph': 'graph_group'
    }

    accuracy_pass1_grouped_structure_model = calculate_grouped_accuracy(combined_data, 1, group_mapping)
    accuracy_pass3_grouped_structure_model = calculate_grouped_accuracy(combined_data, 3, group_mapping)

    accuracy_pass1_grouped_structure_model = accuracy_pass1_grouped_structure_model.reset_index().rename(columns={0: 'Pass@1 Accuracy'})
    accuracy_pass3_grouped_structure_model = accuracy_pass3_grouped_structure_model.reset_index().rename(columns={0: 'Pass@3 Accuracy'})
    accuracy_results_grouped_structure_model = pd.merge(accuracy_pass1_grouped_structure_model, accuracy_pass3_grouped_structure_model, on=['grouped_structure', 'model'])
    
    return accuracy_results_structure_model, accuracy_results_overall_model, accuracy_results_grouped_structure_model
    
path_1 = Path('results/archive/large-macro/openai/openai-zero_shot-large_macro_edit.csv')
path_2 = Path('results/archive/large-macro/anthropic/anthropic-sonnet-zero_shot-large_macro.csv')

#accuracy_by_num_nodes(path_1, path_2)
#accuracy_by_task(path_1, path_2)
##match_similarity_by_variation_num_nodes(path_1)

accuracy_results_structure_model, accuracy_results_overall_model, accuracy_results_grouped_structure_model = calculate_accuracies(path_1, path_2)

print(accuracy_results_structure_model)
print(accuracy_results_overall_model)
print(accuracy_results_grouped_structure_model)