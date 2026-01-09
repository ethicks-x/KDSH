"""
Analyze current classifier performance and identify improvement opportunities.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_performance():
    """Analyze train predictions and feature distributions"""
    print("="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    predictions_path = ROOT / "outputs" / "train_predictions.csv"
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Accuracy: {(df['label'] == df['predicted']).mean():.4f}")
    
    # Confusion matrix breakdown
    print("\n" + "="*60)
    print("CONFUSION MATRIX BREAKDOWN")
    print("="*60)
    
    tp = len(df[(df['label'] == 1) & (df['predicted'] == 1)])
    tn = len(df[(df['label'] == 0) & (df['predicted'] == 0)])
    fp = len(df[(df['label'] == 0) & (df['predicted'] == 1)])
    fn = len(df[(df['label'] == 1) & (df['predicted'] == 0)])
    
    print(f"True Positives (Consistent → Consistent):   {tp}")
    print(f"True Negatives (Contradict → Contradict):   {tn}")
    print(f"False Positives (Contradict → Consistent):  {fp}")
    print(f"False Negatives (Consistent → Contradict):  {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Feature distribution analysis
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTIONS")
    print("="*60)
    
    features = ['delta', 'ppl_ratio', 'baseline_loss', 'primed_loss']
    
    for feature in features:
        print(f"\n{feature}:")
        for label_name, label_val in [("Consistent", 1), ("Contradict", 0)]:
            subset = df[df['label'] == label_val][feature]
            print(f"  {label_name:12s}: mean={subset.mean():7.4f}, std={subset.std():7.4f}, "
                  f"min={subset.min():7.4f}, max={subset.max():7.4f}")
    
    # Book-specific analysis
    print("\n" + "="*60)
    print("BOOK-SPECIFIC PERFORMANCE")
    print("="*60)
    
    for book in df['book_name'].unique():
        book_df = df[df['book_name'] == book]
        accuracy = (book_df['label'] == book_df['predicted']).mean()
        print(f"\n{book}:")
        print(f"  Examples: {len(book_df)}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS - Sample Misclassifications")
    print("="*60)
    
    errors = df[df['label'] != df['predicted']]
    
    if len(errors) > 0:
        print(f"\nTotal errors: {len(errors)}")
        
        # Show a few examples
        for idx, row in errors.head(3).iterrows():
            print(f"\n--- Error #{idx} ---")
            print(f"Character: {row['char']}")
            print(f"Book: {row['book_name']}")
            print(f"True label: {'Consistent' if row['label']==1 else 'Contradict'}")
            print(f"Predicted:  {'Consistent' if row['predicted']==1 else 'Contradict'}")
            print(f"Confidence: {row['confidence']:.3f}")
            print(f"Delta: {row['delta']:.4f}, PPL ratio: {row['ppl_ratio']:.3f}")
    
    # Create visualizations
    create_visualizations(df)
    
    return df


def create_visualizations(df):
    """Create performance visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Delta distribution by label
    plt.figure(figsize=(10, 6))
    
    consistent = df[df['label'] == 1]['delta']
    contradict = df[df['label'] == 0]['delta']
    
    plt.hist(consistent, bins=30, alpha=0.5, label='Consistent', color='green')
    plt.hist(contradict, bins=30, alpha=0.5, label='Contradict', color='red')
    plt.xlabel('Delta (Primed Loss - Baseline Loss)')
    plt.ylabel('Count')
    plt.title('Delta Distribution by Label')
    plt.legend()
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "delta_distribution.png", dpi=150)
    print(f"✅ Saved: delta_distribution.png")
    plt.close()
    
    # 2. PPL ratio distribution
    plt.figure(figsize=(10, 6))
    
    consistent_ppl = df[df['label'] == 1]['ppl_ratio']
    contradict_ppl = df[df['label'] == 0]['ppl_ratio']
    
    plt.hist(consistent_ppl, bins=30, alpha=0.5, label='Consistent', color='green')
    plt.hist(contradict_ppl, bins=30, alpha=0.5, label='Contradict', color='red')
    plt.xlabel('Perplexity Ratio (Primed / Baseline)')
    plt.ylabel('Count')
    plt.title('Perplexity Ratio Distribution by Label')
    plt.legend()
    plt.axvline(1.0, color='black', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "ppl_ratio_distribution.png", dpi=150)
    print(f"✅ Saved: ppl_ratio_distribution.png")
    plt.close()
    
    # 3. Confidence distribution
    plt.figure(figsize=(10, 6))
    
    correct = df[df['label'] == df['predicted']]['confidence']
    incorrect = df[df['label'] != df['predicted']]['confidence']
    
    plt.hist(correct, bins=30, alpha=0.5, label='Correct', color='green')
    plt.hist(incorrect, bins=30, alpha=0.5, label='Incorrect', color='red')
    plt.xlabel('Classifier Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence: Correct vs Incorrect')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "confidence_distribution.png", dpi=150)
    print(f"✅ Saved: confidence_distribution.png")
    plt.close()
    
    # 4. Feature correlation heatmap
    plt.figure(figsize=(8, 6))
    
    features = ['delta', 'ppl_ratio', 'baseline_loss', 'primed_loss', 'label']
    corr = df[features].corr()
    
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "feature_correlation.png", dpi=150)
    print(f"✅ Saved: feature_correlation.png")
    plt.close()


if __name__ == "__main__":
    analyze_performance()

