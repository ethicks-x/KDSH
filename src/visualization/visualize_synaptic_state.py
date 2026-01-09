"""
Visualize how synaptic state σ changes when processing consistent vs contradictory backstories.
This is the key interpretability artifact for the report.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from textpath import TextPath


def extract_attention_weights(model, input_ids):
    """
    Extract attention patterns from LinearAttention.
    This approximates the synaptic connectivity matrix σ.
    """
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Capture the attention-like patterns
        # This depends on BDH's internal structure
        if hasattr(module, 'attention_weights'):
            attention_weights.append(module.attention_weights.detach().cpu())
        elif isinstance(output, tuple) and len(output) > 1:
            # Some attention modules return (output, attention_weights)
            attention_weights.append(output[1].detach().cpu())
    
    # Register hook on LinearAttention
    handle = model.bdh.linear_attn.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove hook
    handle.remove()
    
    return attention_weights


def visualize_state_comparison():
    """
    Compare synaptic states for consistent vs contradictory backstories.
    """
    print("="*60)
    print("SYNAPTIC STATE VISUALIZATION")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Test cases
    print("\nCreating test cases...")
    
    # Novel excerpt (ground truth)
    novel_excerpt = """
    Edmond Dantès was a young sailor from Marseille. He was the first mate
    aboard the Pharaon and was engaged to the beautiful Mercedes. After his
    captain died, Edmond successfully brought the ship to port and was promised
    a promotion to captain by the ship owner Morrel.
    """
    
    # Consistent backstory
    backstory_consistent = """
    Edmond Dantès spent his youth in Marseille, learning the ways of the sea.
    He was known for his honesty and skill as a sailor. His love for Mercedes
    was well known throughout the port.
    """
    
    # Contradictory backstory
    backstory_contradict = """
    Edmond Dantès was a wealthy aristocrat from Paris who had never worked
    as a sailor. He despised the sea and Mercedes was his enemy, not his love.
    """
    
    # Tokenize
    novel_ids = torch.tensor([tokenizer.encode(novel_excerpt).ids[:100]]).to(device)
    consistent_ids = torch.tensor([tokenizer.encode(backstory_consistent).ids[:50]]).to(device)
    contradict_ids = torch.tensor([tokenizer.encode(backstory_contradict).ids[:50]]).to(device)
    
    print(f"\nNovel tokens: {novel_ids.shape[1]}")
    print(f"Consistent backstory tokens: {consistent_ids.shape[1]}")
    print(f"Contradictory backstory tokens: {contradict_ids.shape[1]}")
    
    # === Visualization 1: Activation patterns ===
    print("\nGenerating activation heatmaps...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get hidden states for each input
    with torch.no_grad():
        # Novel alone
        novel_logits, _ = model(novel_ids)
        novel_activations = novel_logits[0].cpu().numpy()
        
        # Consistent backstory + novel
        consistent_combined = torch.cat([consistent_ids, novel_ids], dim=1)
        consistent_logits, _ = model(consistent_combined)
        consistent_activations = consistent_logits[0, consistent_ids.shape[1]:].cpu().numpy()
        
        # Contradictory backstory + novel
        contradict_combined = torch.cat([contradict_ids, novel_ids], dim=1)
        contradict_logits, _ = model(contradict_combined)
        contradict_activations = contradict_logits[0, contradict_ids.shape[1]:].cpu().numpy()
    
    # Plot activation patterns (sample dimensions)
    vmax = max(np.abs(novel_activations).max(), 
               np.abs(consistent_activations).max(),
               np.abs(contradict_activations).max())
    
    sns.heatmap(novel_activations[:50, :100].T, ax=axes[0], cmap='RdBu_r', 
                center=0, vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Activation'})
    axes[0].set_title('Novel Alone (Baseline)')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Output Dimension (sampled)')
    
    sns.heatmap(consistent_activations[:50, :100].T, ax=axes[1], cmap='RdBu_r',
                center=0, vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Activation'})
    axes[1].set_title('After Consistent Backstory')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Output Dimension (sampled)')
    
    sns.heatmap(contradict_activations[:50, :100].T, ax=axes[2], cmap='RdBu_r',
                center=0, vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Activation'})
    axes[2].set_title('After Contradictory Backstory')
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Output Dimension (sampled)')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "synaptic_activation_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: synaptic_activation_comparison.png")
    plt.close()
    
    # === Visualization 2: Activation sparsity ===
    print("\nGenerating sparsity analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Compute sparsity (fraction of near-zero activations)
    threshold = 0.01
    
    novel_sparse = (np.abs(novel_activations) < threshold).mean(axis=1)
    consistent_sparse = (np.abs(consistent_activations) < threshold).mean(axis=1)
    contradict_sparse = (np.abs(contradict_activations) < threshold).mean(axis=1)
    
    axes[0].plot(novel_sparse, label='Novel alone', linewidth=2)
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Sparsity (fraction inactive)')
    axes[0].set_title('Baseline Sparsity')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Target 95%')
    axes[0].legend()
    
    axes[1].plot(consistent_sparse, label='After consistent', color='green', linewidth=2)
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Sparsity (fraction inactive)')
    axes[1].set_title('Consistent Backstory Sparsity')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Target 95%')
    axes[1].legend()
    
    axes[2].plot(contradict_sparse, label='After contradict', color='red', linewidth=2)
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Sparsity (fraction inactive)')
    axes[2].set_title('Contradictory Backstory Sparsity')
    axes[2].grid(alpha=0.3)
    axes[2].axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Target 95%')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / "sparsity_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: sparsity_comparison.png")
    plt.close()
    
    # === Visualization 3: Token-level perplexity ===
    print("\nGenerating token-level perplexity...")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Compute token-level log probabilities
    def get_token_logprobs(logits, target_ids):
        # logits: [seq_len, vocab_size]
        # target_ids: [seq_len]
        log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1)
        token_logprobs = []
        for i, target in enumerate(target_ids[0].cpu().numpy()):
            if i < len(log_probs):
                token_logprobs.append(log_probs[i, target].item())
        return np.array(token_logprobs)
    
    novel_target = novel_ids[:, 1:]
    novel_lp = get_token_logprobs(novel_activations[:-1], novel_target)
    consistent_lp = get_token_logprobs(consistent_activations[:-1], novel_target)
    contradict_lp = get_token_logprobs(contradict_activations[:-1], novel_target)
    
    positions = np.arange(len(novel_lp))
    
    ax.plot(positions, -novel_lp, label='Baseline (novel alone)', linewidth=2, alpha=0.8)
    ax.plot(positions, -consistent_lp, label='After consistent backstory', 
            linewidth=2, alpha=0.8, color='green')
    ax.plot(positions, -contradict_lp, label='After contradictory backstory',
            linewidth=2, alpha=0.8, color='red')
    
    ax.set_xlabel('Token Position in Novel')
    ax.set_ylabel('Negative Log Probability (lower = more confident)')
    ax.set_title('Token-Level Prediction Confidence')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "token_level_confidence.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: token_level_confidence.png")
    plt.close()
    
    print("\n✅ All synaptic visualizations complete!")


if __name__ == "__main__":
    visualize_state_comparison()

