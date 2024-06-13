from pathlib import Path
import pandas as pd
import os
import tiktoken

def analyze_csvs(csv_files):
    results = []
    tokenizer = tiktoken.get_encoding('cl100k_base')

    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Filter out rows with NaN values in the 'response' column
        df = df[df['response'].notna()]

        # Calculate character count
        df['char_count'] = df['response'].apply(len)

        # Calculate word count
        df['word_count'] = df['response'].apply(lambda x: len(x.split()))
        
        df['tokens'] = df['response'].apply(lambda x: tokenizer.encode(str(x)))

        # Extract the model name from the CSV file name
        model_name = os.path.basename(csv_file).split('_')[0]

        # Calculate the total character count
        total_chars = df['char_count'].sum()

        # Calculate the average character count per entry
        avg_chars = df['char_count'].mean()

        # Calculate the standard deviation of character count per entry
        std_dev_chars = df['char_count'].std()

        # Calculate the minimum and maximum character count per entry
        min_chars = df['char_count'].min()
        max_chars = df['char_count'].max()

        # Calculate the total word count
        total_words = df['word_count'].sum()

        # Calculate the average word count per entry
        avg_words = df['word_count'].mean()

        # Calculate the standard deviation of word count per entry
        std_dev_words = df['word_count'].std()

        # Calculate the minimum and maximum word count per entry
        min_words = df['word_count'].min()
        max_words = df['word_count'].max()
        
        # Calculate the total number of tokens
        total_tokens = df['tokens'].apply(len).sum()

        # Calculate the average tokens per entry
        avg_tokens = df['tokens'].apply(len).mean()

        # Calculate the standard deviation of tokens per entry
        std_dev_tokens = df['tokens'].apply(len).std()

        # Calculate the minimum and maximum tokens per entry
        min_tokens = df['tokens'].apply(len).min()
        max_tokens = df['tokens'].apply(len).max()

        # Create a dictionary with the calculated metrics
        result = {
            'Model': model_name,
            'Total Characters': total_chars,
            'Average Characters': round(avg_chars),
            'Std Dev Characters': round(std_dev_chars),
            'Min Characters': min_chars,
            'Max Characters': max_chars,
            'Total Words': total_words,
            'Average Words': round(avg_words),
            'Std Dev Words': round(std_dev_words),
            'Min Words': min_words,
            'Max Words': max_words,
            'Total Tokens': total_tokens,
            'Average Tokens': round(avg_tokens),
            'Standard Deviation': round(std_dev_tokens),
            'Minimum Tokens': min_tokens,
            'Maximum Tokens': max_tokens,
        }
        results.append(result)

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)
    return result_df

path_1 = Path('results/archive/large-macro/openai/openai-zero_shot-large_macro_edit.csv')
path_2 = Path('results/archive/large-macro/deepmind/1.5/deepmind-15-zero_shot-large_macro.csv')
path_3 = Path('results/archive/large-macro/deepmind/1.0/deepmind-zero_shot-large_macro.csv')
path_4 = Path('results/archive/large-macro/anthropic/opus/anthropic-zero_shot-large_macro.csv')
path_5 = Path('results/archive/large-macro/anthropic/sonnet/anthropic-sonnet-zero_shot-large_macro.csv')
path_6 = Path('results/archive/large-macro/anthropic/haiku/anthropic-haiku-zero_shot-large_macro.csv')

paths = [path_1, path_2, path_3, path_4, path_5, path_6]

result_table = analyze_csvs(paths)
# export to csv
result_table.to_csv('tokens.csv', index=False)
pd.set_option('display.max_colwidth', None)
print(result_table)