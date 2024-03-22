import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score, mean_squared_error
from scipy.stats import t, ttest_1samp, norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
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

def read_data(csv_path: Path):
    return pd.read_csv(csv_path)

def plot_actual_vs_predicted(y_actual, y_predicted):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_actual, y=y_predicted, alpha=0.5, label='Predictions')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
    # set spine color
    # set grid color
    plt.grid(color='lightgray')
    #plt.suptitle(f'{CSV_PATH.stem.replace("_", "-")}', fontproperties=sohne_bold_font, fontsize=16, x=0.13, y=0.96, ha='left')
    plt.legend(loc="lower right")
    plt.xlabel('Actual Similarity')
    plt.ylabel('Predicted Similarity')
    plt.title('Actual vs. Predicted Similarity', loc='left', fontproperties=sohne_bold_font, fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("plot/actual_vs_predicted.png", dpi=300)
    
def plot_residuals(y_actual, y_predicted):
    residuals = y_actual - y_predicted
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_predicted, y=residuals, alpha=0.6, label='Residuals')
    # set grid color
    #plt.grid(color='lightgray')
    #plt.suptitle(f'{CSV_PATH.stem.replace("_", "-")}', fontproperties=sohne_bold_font, fontsize=16, x=0.122, y=0.96, ha='left')
    plt.axhline(y=0, color='r', linestyle='--', label='No Residual')
    plt.xlabel('Predicted Similarity')
    plt.ylabel('Residuals')
    plt.legend(loc="upper right")
    plt.title('Residuals of Predicted Similarity', loc='left', fontproperties=sohne_bold_font, fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig("plot/residuals.png", dpi=300)

def plot_roc_curve(y_actual, y_score):
    fpr, tpr, thresholds = roc_curve(y_actual, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve\nAUC %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level\nAUC 0.50')
    # set grid color
    #plt.grid(color='lightgray')
    #plt.suptitle(f'{CSV_PATH.stem.replace("_", "-")}', fontproperties=sohne_bold_font, fontsize=16, x=0.122, y=0.96, ha='left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Match Prediction', loc='left', fontproperties=sohne_bold_font, fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("plot/roc_curve.png", dpi=300)
    
def plot_confusion_matrix(y_actual, y_pred, class_names):
    matrix = confusion_matrix(y_actual, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    #plt.suptitle(f'{CSV_PATH.stem.replace("_", "-")}', fontproperties=sohne_bold_font, fontsize=16, x=0.15, y=0.88, ha='left')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Match Prediction', loc='left', fontproperties=sohne_bold_font, fontsize=12)
    plt.tight_layout()
    plt.savefig("plot/confusion_matrix.png", dpi=300)
    
def plot_precision_recall_curve(y_actual, y_score):
    precision, recall, _ = precision_recall_curve(y_actual, y_score)
    avg_precision = average_precision_score(y_actual, y_score)
    
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall Curve\nAvg. Precision: {avg_precision:.2f}')
    plt.plot([0, 1], [0.5, 0.5], color='navy', lw=2, linestyle='--', label='Chance Level')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Match Prediction', loc='left', fontproperties=sohne_bold_font, fontsize=12)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("plot/precision_recall_curve.png", dpi=300)
    
def plot_feature_correlation(X, feature_names):
    feature_names = [truncate_name(name).replace("_", " ") for name in feature_names]
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    # remove minus sign
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(45, 40))
    # reverse the cmap
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm_r", linewidths=0.1)
    plt.title("Feature Correlation Heatmap", fontproperties=sohne_bold_font, fontsize=24, loc='left')
    plt.tick_params(axis='both', which='major', labelsize=12)
    #plt.tight_layout()
    plt.savefig("plot/feature_correlation_heatmap.png", dpi=100)

def plot_tsne(X, y_class, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot t-SNE visualization for the entire dataset
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_class, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title("t-SNE Feature Space", fontproperties=sohne_bold_font, fontsize=12, loc='left')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig("plot/tsne_plot.png", dpi=300)
    
def regression_confidence_intervals(y_test, y_pred, alpha=0.05):
    n = len(y_test)
    mse = mean_squared_error(y_test, y_pred)
    se = np.sqrt(mse / n)
    t_value = t.ppf(1 - alpha / 2, n - 1)
    lower = np.mean(y_pred) - t_value * se
    upper = np.mean(y_pred) + t_value * se
    return lower, upper

def classification_confidence_intervals(y_test, y_pred, alpha=0.05):
    n = len(y_test)
    p = accuracy_score(y_test, y_pred)
    se = np.sqrt(p * (1 - p) / n)
    z_value = 1.96  # Assuming a large sample size and 95% confidence level
    lower = p - z_value * se
    upper = p + z_value * se
    return lower, upper
    
def print_model_specifications(model):
    """
    Prints specifications of a given model. Works with standalone models
    and sklearn pipelines.
    """
    if hasattr(model, 'steps'):  # It's a Pipeline
        print("Model is a Pipeline. Steps:")
        for step_name, step_process in model.steps:
            print(f"\nStep: {step_name}")
            print(f"Process: {step_process}")
            
            # If the step is a transformer or estimator with parameters, print them
            if hasattr(step_process, 'get_params'):
                print("Parameters:")
                params = step_process.get_params()
                for param_name, param_value in params.items():
                    print(f"    {param_name}: {param_value}")
    else:
        # Standalone model, just print its parameters
        print("Model Specifications:")
        print(model)
        
        if hasattr(model, 'get_params'):
            print("Parameters:")
            params = model.get_params()
            for param_name, param_value in params.items():
                print(f"    {param_name}: {param_value}")
                
def regression_hypothesis_test(y_test, y_pred, baseline_mse, alpha=0.05):
    mse = mean_squared_error(y_test, y_pred)
    n = len(y_test)
    se = np.sqrt(mse / n)
    t_statistic, p_value = ttest_1samp(y_pred, baseline_mse)
    
    print(f"Regression Hypothesis Test:")
    print(f"Null Hypothesis: Model MSE = {baseline_mse}")
    print(f"Alternative Hypothesis: Model MSE != {baseline_mse}")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("Reject the null hypothesis. The model's performance is significantly different from the baseline.")
    else:
        print("Fail to reject the null hypothesis. The model's performance is not significantly different from the baseline.")

def classification_hypothesis_test(y_test, y_pred, baseline_accuracy, alpha=0.05):
    accuracy = accuracy_score(y_test, y_pred)
    n = len(y_test)
    
    z_statistic = (accuracy - baseline_accuracy) / np.sqrt((baseline_accuracy * (1 - baseline_accuracy)) / n)
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))
    
    print(f"Classification Hypothesis Test:")
    print(f"Null Hypothesis: Model Accuracy = {baseline_accuracy}")
    print(f"Alternative Hypothesis: Model Accuracy != {baseline_accuracy}")
    print(f"z-statistic: {z_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("Reject the null hypothesis. The model's performance is significantly different from the baseline.")
    else:
        print("Fail to reject the null hypothesis. The model's performance is not significantly different from the baseline.")

def plot_pca_variance(data):
    pca = PCA().fit(data)  # `data` is your feature matrix
    plt.figure(figsize=(5, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # label legend
    plt.title('PCA Variance Explained', fontproperties=sohne_bold_font, fontsize=12, loc='left')
    plt.xlabel('Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.tight_layout()
    plt.savefig("plot/pca_variance.png", dpi=100)
    
def truncate_name(name, max_length=20):
    """Truncate or pad a string to a specific length."""
    if len(name) > max_length:
        return name[:max_length-3] + "..."
    else:
        return name

def plot_feature_importances_with_model(feature_names, model, title):
    """
    Plot the top N feature importances for a model (regression or classification) with signs and truncated labels.
    
    Parameters:
    - feature_names: list of feature names
    - model: the model object (sklearn model or pipeline with 'regressor' or 'classifier' step)
    - title: title for the plot
    """
    # Extract coefficients from the model
    if hasattr(model, 'named_steps'):
        if 'regressor' in model.named_steps:
            model_coefs = model.named_steps['regressor'].coef_
        elif 'classifier' in model.named_steps:
            model_coefs = model.named_steps['classifier'].coef_[0]  # Assuming binary classification or one-vs-rest
        else:
            raise ValueError("The model provided does not have 'regressor' or 'classifier' steps with accessible coefficients.")
    elif hasattr(model, 'coef_'):
        model_coefs = model.coef_[0]  # Assuming binary classification or one-vs-rest
    else:
        raise ValueError("The model provided does not have coefficients accessible via 'coef_'.")

    # Plot the feature importances for the model
    plt.figure(figsize=(5, 15))
    sorted_idx = np.argsort(np.abs(model_coefs))[::-1]
    top_n = len(feature_names)  # Show top 30 features for clarity
    plt.barh(range(top_n), model_coefs[sorted_idx[:top_n]], color='cadetblue', align='center')
    truncated_feature_names = [name.replace("_", " ") if len(name) <= 20 else name[:17].replace("_", " ").replace("'", "") + "..." for name in np.array(feature_names)[sorted_idx[:top_n]]]
    plt.yticks(range(top_n), truncated_feature_names)
    plt.gca().invert_yaxis()  # Display the highest importance at the top
    plt.title(title, fontproperties=sohne_bold_font, fontsize=10, loc='left', pad=5)
    # change y tick font size
    plt.yticks(fontsize=5)
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('Coefficient Magnitude', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    plt.grid(True, which='major', linestyle='--', linewidth=0.1)
    plt.tight_layout()
    plt.savefig(f"plot/feature_importances_{title}.png", dpi=300)
    
def plot_3d_pca(pca_image_features, y_class):
    X_pca_3d = pca_image_features[:, :3]

    # Plotting
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Coloring by the classification target for better visual distinction
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y_class, cmap='viridis', alpha=0.5)

    # Adding color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Class')

    ax.set_title('PCA Image Feature Space', fontproperties=sohne_bold_font, fontsize=12, loc='left')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    plt.tight_layout()
    plt.savefig("plot/3d_pca.png", dpi=300)
    
def plot_pca_loadings_heatmap(pca, feature_names, n_components=3):
    loadings = pca.components_[:n_components, :]
    
    loadings_df = pd.DataFrame(loadings[:, :len(feature_names)], columns=feature_names, index=[f'PC{i+1}' for i in range(n_components)])
    truncated_feature_names = [truncate_name(name) for name in feature_names]
    
    plt.figure(figsize=(10, 20))
    sns.heatmap(loadings_df.T, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title(f'PCA Loadings Heatmap for First {n_components} Principal Components', fontproperties=sohne_bold_font, fontsize=12, loc='left')
    plt.ylabel('Original Features', fontsize=14)
    plt.xlabel('Principal Components', fontsize=14)
    # truncate y axis labels
    plt.yticks(ticks=np.arange(len(feature_names)), labels=truncated_feature_names, fontsize=8)
    plt.tight_layout()
    plt.savefig("plot/pca_loadings_heatmap.png", dpi=300)

def main(DATA_PATH, CSV_PATH):
    
    df = read_data(CSV_PATH)
    
    metrics_reg = load(DATA_PATH / Path('regression_metrics.joblib'))
    metrics_class = load(DATA_PATH / Path('classification_metrics.joblib'))
    model_reg = load(DATA_PATH / Path('regression_model.joblib'))
    model_class = load(DATA_PATH / Path('classification_model.joblib'))
    feature_names = load(DATA_PATH / Path('feature_names.joblib')) # Load the feature names
    X = load(DATA_PATH / Path('X.joblib')) # Load the feature matrix
    svd = load(DATA_PATH / Path('svd.joblib')) # Load the SVD model
    tfidf_vectorizer = load(DATA_PATH / Path('tfidf_vectorizer.joblib')) # Load the TF-IDF vectorizer
    text_prompt_tfidf_svd = load(DATA_PATH / Path('text_prompt_tfidf_svd.joblib')) # Load the SVD-transformed TF-IDF vectors for the text prompt
    pca = load(DATA_PATH / Path('pca_image_model.joblib'))
    pca_image_features = load(DATA_PATH / Path('pca_image_features.joblib'))
    y_class = load(DATA_PATH / Path('y_class.joblib'))
    original_to_pca_index_map = load(DATA_PATH / Path('original_to_pca_index_map.joblib'))
    inverted_map = {v: k for k, v in original_to_pca_index_map.items()}
    y_class_aligned = y_class.reindex(index=pd.Index(inverted_map.keys())).values
    
    if hasattr(model_reg, 'named_steps'):
        regression_coefs = model_reg.named_steps['regressor'].coef_
    else:
        regression_coefs = model_reg.coef_

    if hasattr(model_class, 'named_steps'):
        classification_coefs = model_class.named_steps['classifier'].coef_[0]
    else:
        classification_coefs = model_class.coef_[0]

    print("Shape of X:", X.shape)
    print("Number of feature names:", len(feature_names))

    if X.shape[1] != len(feature_names):
        print("Mismatch between data shape and feature names!")
        # Update feature_names to match the number of columns in X
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        print("Updated feature names:", feature_names)
        
    mean_target = np.mean(metrics_reg['y_test'])
    median_target = np.median(metrics_reg['y_test'])

    # Predictions by the mean and median predictor are just the mean and median values repeated for each test instance
    mean_predictions = np.full_like(metrics_reg['y_test'], fill_value=mean_target)
    median_predictions = np.full_like(metrics_reg['y_test'], fill_value=median_target)

    # Calculate MSE for the mean and median predictor
    baseline_mse_mean = mean_squared_error(metrics_reg['y_test'], mean_predictions)
    baseline_mse_median = mean_squared_error(metrics_reg['y_test'], median_predictions)
    
    #baseline_accuracy = accuracy_score(metrics_class['y_test'], metrics_class['y_test'].value_counts().idxmax())
    
    print(baseline_mse_mean, baseline_mse_median)
    
    # Plot the results
    plot_pca_variance(X)
    plot_actual_vs_predicted(metrics_reg['y_test'], metrics_reg['y_pred'])
    plot_residuals(metrics_reg['y_test'], metrics_reg['y_pred'])
    plot_roc_curve(metrics_class['y_test'], metrics_class['y_pred_prob'])
    plot_confusion_matrix(metrics_class['y_test'], metrics_class['y_pred'], ['No Match', 'Match'])
    #plot_precision_recall_curve(metrics_class['y_test'], metrics_class['y_pred_prob'])
    #plot_feature_correlation(X, feature_names)
    #plot_tsne(X, y_class=y_class, perplexity=30)
    plot_feature_importances_with_model(feature_names, model_reg, "Feature Importances for Similarity Prediction")
    plot_feature_importances_with_model(feature_names, model_class, "Feature Importances for Match Prediction")
    
    plot_3d_pca(pca_image_features, y_class_aligned)
    plot_pca_loadings_heatmap(pca, feature_names, n_components=3)
    
    #print(regression_confidence_intervals(metrics_reg['y_test'], metrics_reg['y_pred']))
    #print(classification_confidence_intervals(metrics_class['y_test'], metrics_class['y_pred']))
    
    #regression_hypothesis_test(metrics_reg['y_test'], metrics_reg['y_pred'], baseline_mse_mean)
    #classification_hypothesis_test(metrics_class['y_test'], metrics_class['y_pred'], baseline_accuracy)
    
    #print_model_specifications(model_reg)
    #print_model_specifications(model_class)

if __name__ == "__main__":
    print("Script execution started.")
    # Provide the path to your CSV file here
    DATA_PATH = Path('results/archive/large-course-plus')
    CSV_PATH = Path('results/archive/large-course-plus/deepmind-zero_shot-large_course_plus.csv')
    main(DATA_PATH, CSV_PATH)
    print("Script execution completed.")