import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from uuid import uuid4
import os
from matplotlib.font_manager import FontProperties, fontManager

signifier_font_path = "plot/fonts/Test Signifier/TestSignifier-Medium.otf"
sohne_font_path = "plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Buch.otf"
sohne_bold_font_path = "plot/fonts/Test Söhne Collection/Test Söhne/TestSöhne-Kräftig.otf"

signifier_font = FontProperties(fname=signifier_font_path)
sohne_font = FontProperties(fname=sohne_font_path)
sohne_bold_font = FontProperties(fname=sohne_bold_font_path)

signifier_font_name = signifier_font.get_name()
sohne_font_name = sohne_font.get_name()
sohne_bold_font_name = sohne_bold_font.get_name()

# Register the fonts with Matplotlib's font manager
fontManager.addfont(signifier_font_path)
fontManager.addfont(sohne_font_path)
fontManager.addfont(sohne_bold_font_path)

plt.rcParams['font.family'] = sohne_font_name

# Load the dataset
file_path = Path('results/deepmind-prompts_default.csv')

def match_similarity_per_structure_grouped_by_num_nodes(file_path: Path) -> None:
    df = pd.read_csv(file_path)
    
    # Calculate overall match rate and average similarity
    overall_match_rate = df['match'].mean()
    overall_average_similarity = df['similarity'].mean()
    df['similarity'] = df['similarity'] / 100
    
    # Grouping by 'structure' and 'num_nodes' to calculate match rate and average similarity
    grouped_data = df.groupby(['structure', 'num_nodes']).agg(match_rate=('match', 'mean'), average_similarity=('similarity', 'mean')).reset_index()

    # Setting the plot size a bit larger to accommodate legends and titles better
    plt.figure(figsize=(16, 8))

    # Match Rate Visualization
    ax1 = plt.subplot(1, 2, 1)
    match_plot = sns.barplot(x='num_nodes', y='match_rate', hue='structure', data=grouped_data, palette='coolwarm', ax=ax1)
    #plt.title('Accuracy of Predicted vs. Ground Truth', fontsize=12, loc='left')
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    for spine in ax1.spines.values():
        spine.set_visible(False)
    leg = match_plot.legend(loc='upper right', bbox_to_anchor=(1.01, 1))
    #leg.set_title('Structure', prop=sohne_font)
    for text in leg.get_texts():
        text.set_text(text.get_text().replace("_", " "))
    ax1.set_axisbelow(True)
    ax1.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')

    # Average Similarity Visualization
    ax2 = plt.subplot(1, 2, 2)
    similarity_plot = sns.barplot(x='num_nodes', y='average_similarity', hue='structure', data=grouped_data, palette='coolwarm', ax=ax2)
    #plt.title('Similarity of Predicted vs. Ground Truth', fontsize=12, loc='left')
    plt.xlabel('number of nodes')
    plt.ylabel('similarity')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    leg = similarity_plot.legend(loc='upper right', bbox_to_anchor=(1.01, 1))
    #leg.set_title('Structure', prop=sohne_font)
    for text in leg.get_texts():
        text.set_text(text.get_text().replace("_", " "))
    ax2.set_axisbelow(True)
    ax2.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')
    
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # Display overall average similarity
    plt.figtext(0.83, 0.92, f'Aggregate Mean Similarity - {overall_average_similarity:.2f}', ha='left', fontsize=10, color='red')
    # Display overall match rate
    plt.figtext(0.69, 0.92, f'Aggregate Mean Accuracy - {overall_match_rate:.2f}', ha='left', fontsize=10, color='red')

    # Enhancing the suptitle formatting
    plt.figtext(0.05, 0.92, f'{((file_path.name).replace("_", "-")).replace(".csv", "")}', va='center', fontsize=32, fontweight='bold', color='black')

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
    plt.savefig(f'plot/comparison_{file_path1.stem}_vs_{file_path2.stem}.png', dpi=300)
    

def match_similarity_by_variation_num_nodes(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    #sns.set_style("whitegrid")

    data['similarity'] = data['similarity'] / 100
    structures = data['structure'].unique()
    num_structures = len(structures)

    fig, axes = plt.subplots(2, num_structures, figsize=(12, 6), dpi=300, sharex=True, sharey=True)
    #fig.suptitle("Accuracy and Similarity by Number of Nodes", fontsize=16, y=0.95)

    for i, structure in enumerate(structures):
        structure_data = data[data['structure'] == structure]
        summary = structure_data.groupby(['generation_id', 'variation_id', 'num_nodes']).agg(
            mean_similarity=('similarity', 'mean'),
            match_rate=('match', 'mean')
        ).reset_index()

        sns.lineplot(ax=axes[0, i], data=summary, x='num_nodes', y='match_rate', marker='o', lw=2.0, ms=6.0)
        axes[0, i].set_title(f"{structure.replace('_', ' ')}", loc='left')
        axes[0, i].set_xlabel('n nodes')
        axes[0, i].set_ylabel('accuracy')

        sns.lineplot(ax=axes[1, i], data=summary, x='num_nodes', y='mean_similarity', marker='o')
        axes[1, i].set_xlabel('n nodes')
        axes[1, i].set_ylabel('similarity')
        
        for ax in axes.flat:
            # Hide top and right spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color((0, 0, 0, 0.2))
            ax.spines['bottom'].set_color((0, 0, 0, 0.2))
            
        axes[0, i].set_xticks([3, 6, 9])
        axes[1, i].set_xticks([3, 6, 9])

    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Variation per Number of Nodes with Standard Deviation', fontsize=12, x=0.05, y=0.90, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('plot/match_similarity_by_variation_num_nodes.png')
        
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
        
    plt.savefig(f'plot/match_similarity_by_arrow_style.png', dpi=300)
    
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
        
    plt.savefig(f'plot/match_similarity_by_color.png', dpi=300)
    
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
        
    plt.savefig(f'plot/match_similarity_by_width.png', dpi=300)
    
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

    plt.savefig(f'plot/heatmap_of_match_and_similarity_by_width_and_color.png', dpi=300)
    
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
    plt.savefig(f'plot/match_similarity_by_edge_width.png', dpi=300)


def line_plot_match_similarity_by_num_nodes(file_path: Path) -> None:
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
    
    fig.text(0.05, 0.92, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Number of Nodes with Standard Deviation', fontsize=12, x=0.05, y=0.90, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f'plot/match_similarity_by_num_nodes.png', dpi=300)


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
    plt.savefig(f'plot/match_similarity_by_color.png', dpi=300)

def bar_plot_match_similarity_by_task(file_path: Path) -> None:
    data = pd.read_csv(file_path)
    data['similarity'] = data['similarity'] / 100
    data['structure'] = data['structure'].replace('_', ' ', regex=True)
    data['task'] = data['task'].str.replace('_', ' ')

    # Define the task groups
    task_groups = {
        'Trees': ['post order', 'pre order', 'in order'],
        'Graphs': ['breadth first search', 'depth first search', 'adjacency list']
    }

    # Create a figure with separate subplots for each task group
    fig, axes = plt.subplots(2, len(task_groups), figsize=(10, 10), sharex=False, sharey=True)

    for i, (group_name, tasks) in enumerate(task_groups.items()):
        # Filter the data for the current task group
        group_data = data[data['task'].isin(tasks)]

        # Grouping by 'task' and 'structure' for the match plot
        sns.barplot(data=group_data, x='task', y='match', hue='structure', estimator=np.mean, ci='sd', ax=axes[0, i], palette='coolwarm')
        axes[0, i].set_ylabel('accuracy')
        axes[0, i].set_xlabel('')
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_xticklabels([])

        # Grouping by 'task' and 'structure' for the similarity plot
        sns.barplot(data=group_data, x='task', y='similarity', hue='structure', estimator=np.mean, ci='sd', ax=axes[1, i], palette='coolwarm')
        axes[1, i].set_ylabel('similarity')
        axes[1, i].set_xlabel('task')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].get_legend().remove()
        
        # remove spines
        for ax in axes.flat:
            # Hide top and right spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color((0, 0, 0, 0.2))
            ax.spines['bottom'].set_color((0, 0, 0, 0.2))
            
    axes[0, 0].legend(title='structure', loc='upper right')
    axes[0, 1].legend(title='structure', loc='upper left')

    fig.text(0.07, 0.95, file_path.stem.replace('_', '-'), fontsize=24, ha='left', fontproperties=sohne_bold_font)
    fig.suptitle('Mean Predicted vs Ground Truth Performance by Task', fontsize=12, x=0.07, y=0.93, ha='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'plot/match_similarity_by_task_grouped.png', dpi=300)

match_similarity_per_structure_grouped_by_num_nodes(Path('results/deepmind-zero_shot-large_course.csv'))
heatmap_match_similarity_by_width_and_color(Path('results/deepmind-zero_shot-large_course.csv'))
line_plot_match_similarity_by_edge_width(Path('results/deepmind-zero_shot-large_course.csv'))
line_plot_match_similarity_by_num_nodes(Path('results/deepmind-zero_shot-large_course.csv'))
#line_plot_match_similarity_by_variation_generation(Path('results/deepmind-zero_shot-large_course.csv'))
bar_plot_match_similarity_by_color(Path('results/deepmind-zero_shot-large_course.csv'))
bar_plot_match_similarity_by_task(Path('results/deepmind-zero_shot-large_course.csv'))
match_similarity_by_variation_generation(Path('results/deepmind-zero_shot-large_course.csv'))
match_similarity_by_variation_num_nodes(Path('results/deepmind-zero_shot-large_course.csv'))