"""
Tune hyperparameters for consistency detection.
Tests: retrieval K, chunk size, feature engineering.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "data_processing"))

from textpath import TextPath
from consistency_scorer import ConsistencyScorer
from retrieval import NovelRetriever


def tune_retrieval_k():
    """Test different values of top-K retrieval"""
    print("="*60)
    print("TUNING: Retrieval K")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    
    # Load resources
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    train_csv = ROOT / "Dataset" / "train.csv"
    books_dir = ROOT / "Dataset" / "Books"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    scorer = ConsistencyScorer(model, tokenizer, device, max_novel_tokens=512)
    
    # Build retrievers
    retrievers = {
        "The Count of Monte Cristo": NovelRetriever(
            books_dir / "The Count of Monte Cristo.txt",
            chunk_size=400,
            overlap=100
        ),
        "In Search of the Castaways": NovelRetriever(
            books_dir / "In search of the castaways.txt",
            chunk_size=400,
            overlap=100
        ),
    }
    
    df = pd.read_csv(train_csv)
    
    # Map labels
    df['label_int'] = df['label'].map({'consistent': 1, 'contradict': 0})
    df['label_int'] = df['label_int'].fillna(df['label']).astype(int)
    
    # Test different K values
    k_values = [1, 3, 5, 7, 10]
    results = []
    
    for k in k_values:
        print(f"\nTesting K={k}...")
        
        # Score all examples with this K
        all_scores = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"K={k}"):
            retriever = retrievers[row['book_name']]
            chunks = retriever.retrieve(row['content'], top_k=k)
            combined_novel = ' '.join([chunk for chunk, _ in chunks])
            
            scores = scorer.score_consistency(row['content'], combined_novel)
            all_scores.append(scores)
        
        # Create feature matrix
        X = np.array([[s['delta'], s['ppl_ratio'], s['baseline_loss'], s['primed_loss']] 
                      for s in all_scores])
        y = df['label_int'].values
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)
        
        # Evaluate
        train_acc = accuracy_score(y, clf.predict(X))
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        results.append({
            'k': k,
            'train_accuracy': train_acc,
            'cv_accuracy': cv_acc,
            'cv_std': cv_scores.std()
        })
        
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  CV Acc:    {cv_acc:.4f} (+/- {cv_scores.std():.4f})")
    
    # Print summary
    print("\n" + "="*60)
    print("RETRIEVAL K TUNING RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_k = results_df.loc[results_df['cv_accuracy'].idxmax(), 'k']
    print(f"\n✅ Best K: {best_k}")
    
    # Save results
    output_path = ROOT / "outputs" / "tuning_retrieval_k.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")
    
    return best_k, results_df


def tune_chunk_size():
    """Test different chunk sizes for retrieval"""
    print("\n" + "="*60)
    print("TUNING: Chunk Size")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    train_csv = ROOT / "Dataset" / "train.csv"
    books_dir = ROOT / "Dataset" / "Books"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    scorer = ConsistencyScorer(model, tokenizer, device, max_novel_tokens=512)
    
    df = pd.read_csv(train_csv)
    df['label_int'] = df['label'].map({'consistent': 1, 'contradict': 0})
    df['label_int'] = df['label_int'].fillna(df['label']).astype(int)
    
    # Test different chunk sizes
    chunk_sizes = [200, 300, 400, 500, 600]
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk_size={chunk_size}...")
        
        # Build retrievers with this chunk size
        retrievers = {
            "The Count of Monte Cristo": NovelRetriever(
                books_dir / "The Count of Monte Cristo.txt",
                chunk_size=chunk_size,
                overlap=100
            ),
            "In Search of the Castaways": NovelRetriever(
                books_dir / "In search of the castaways.txt",
                chunk_size=chunk_size,
                overlap=100
            ),
        }
        
        # Score subset (faster)
        sample_df = df.sample(min(50, len(df)), random_state=42)
        all_scores = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"chunk={chunk_size}"):
            retriever = retrievers[row['book_name']]
            chunks = retriever.retrieve(row['content'], top_k=3)
            combined_novel = ' '.join([chunk for chunk, _ in chunks])
            
            scores = scorer.score_consistency(row['content'], combined_novel)
            all_scores.append(scores)
        
        X = np.array([[s['delta'], s['ppl_ratio']] for s in all_scores])
        y = sample_df['label_int'].values
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        
        results.append({
            'chunk_size': chunk_size,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        print(f"  CV Acc: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\n" + "="*60)
    print("CHUNK SIZE TUNING RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_chunk = results_df.loc[results_df['cv_accuracy'].idxmax(), 'chunk_size']
    print(f"\n✅ Best chunk_size: {best_chunk}")
    
    output_path = ROOT / "outputs" / "tuning_chunk_size.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_chunk, results_df


def main():
    """Run all hyperparameter tuning"""
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Tune retrieval K
    best_k, k_results = tune_retrieval_k()
    
    # Tune chunk size (optional, takes longer)
    # best_chunk, chunk_results = tune_chunk_size()
    
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Optimal retrieval K: {best_k}")
    # print(f"Optimal chunk size: {best_chunk}")
    
    # Save final config
    ROOT = Path(__file__).resolve().parents[2]
    config = {
        'retrieval_k': int(best_k),
        'chunk_size': 400,  # Default or from chunk tuning
        'overlap': 100,
        'max_novel_tokens': 512,
    }
    
    with open(ROOT / "outputs" / "optimal_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Optimal config saved to outputs/optimal_config.json")


if __name__ == "__main__":
    main()

