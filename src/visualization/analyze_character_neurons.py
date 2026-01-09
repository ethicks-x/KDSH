"""
Analyze if specific neurons correspond to character concepts (hub neurons).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tokenizers import Tokenizer
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from textpath import TextPath


def analyze_character_neurons():
    """
    Test if model has learned character-specific neurons.
    """
    print("="*60)
    print("CHARACTER NEURON ANALYSIS")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Test sentences mentioning different characters
    test_sentences = {
        'Dantès': [
            "Edmond Dantès was arrested",
            "Dantès escaped from prison",
            "The count greeted Dantès warmly"
        ],
        'Mercedes': [
            "Mercedes waited for Edmond",
            "Mercedes married Fernand",
            "The beautiful Mercedes wept"
        ],
        'Villefort': [
            "Villefort was the prosecutor",
            "The judge Villefort decided",
            "Villefort hid the letter"
        ],
        'Thalcave': [
            "Thalcave guided the expedition",
            "The Patagonian Thalcave helped",
            "Thalcave tracked the trail"
        ],
        'Glenarvan': [
            "Lord Glenarvan organized the search",
            "Glenarvan owned the Duncan",
            "The noble Glenarvan led"
        ]
    }
    
    # Get activations for each character's sentences
    print("\nExtracting activations...")
    
    character_activations = defaultdict(list)
    
    for char_name, sentences in test_sentences.items():
        print(f"  Processing {char_name}...")
        for sentence in sentences:
            ids = torch.tensor([tokenizer.encode(sentence).ids]).to(device)
            
            with torch.no_grad():
                logits, _ = model(ids)
            
            # Average activations across sequence
            avg_activation = logits[0].mean(dim=0).cpu().numpy()
            character_activations[char_name].append(avg_activation)
    
    # Average across sentences for each character
    character_profiles = {}
    for char_name, activations in character_activations.items():
        character_profiles[char_name] = np.mean(activations, axis=0)
    
    # Find most distinctive dimensions for each character
    print("\nFinding character-specific neurons...")
    
    all_profiles = np.array([character_profiles[c] for c in character_profiles.keys()])
    char_names = list(character_profiles.keys())
    
    # For each dimension, compute variance across characters
    # High variance = dimension discriminates between characters
    dimension_variance = np.var(all_profiles, axis=0)
    top_dims = np.argsort(dimension_variance)[-20:]  # Top 20 discriminative dimensions
    
    print(f"\nTop discriminative dimensions: {top_dims}")
    
    # Visualize character profiles on these dimensions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(top_dims))
    width = 0.15
    
    for i, char_name in enumerate(char_names):
        profile = character_profiles[char_name][top_dims]
        ax.bar(x + i*width, profile, width, label=char_name, alpha=0.8)
    
    ax.set_xlabel('Discriminative Neuron Index')
    ax.set_ylabel('Average Activation')
    ax.set_title('Character-Specific Neuron Activation Profiles')
    ax.set_xticks(x + width * len(char_names) / 2)
    ax.set_xticklabels([f"N{d}" for d in top_dims], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "character_neuron_profiles.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: character_neuron_profiles.png")
    plt.close()
    
    # Compute character similarity matrix
    print("\nComputing character similarities...")
    
    similarity_matrix = np.zeros((len(char_names), len(char_names)))
    
    for i, char1 in enumerate(char_names):
        for j, char2 in enumerate(char_names):
            # Cosine similarity
            p1 = character_profiles[char1]
            p2 = character_profiles[char2]
            similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            similarity_matrix[i, j] = similarity
    
    # Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(char_names)))
    ax.set_yticks(np.arange(len(char_names)))
    ax.set_xticklabels(char_names, rotation=45, ha='right')
    ax.set_yticklabels(char_names)
    
    # Add values
    for i in range(len(char_names)):
        for j in range(len(char_names)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Character Representation Similarity\n(Higher = More Similar Contexts)')
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "character_similarity_matrix.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: character_similarity_matrix.png")
    plt.close()
    
    # Build character co-occurrence network
    print("\nBuilding character network...")
    
    G = nx.Graph()
    for char in char_names:
        G.add_node(char)
    
    # Add edges weighted by similarity
    for i, char1 in enumerate(char_names):
        for j, char2 in enumerate(char_names):
            if i < j and similarity_matrix[i, j] > 0.7:  # Threshold
                G.add_edge(char1, char2, weight=similarity_matrix[i, j])
    
    # Visualize network
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, k=2, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=3000, alpha=0.9, ax=ax)
    
    # Draw edges with thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                          alpha=0.5, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title('Character Network (Based on Neural Representations)\nEdge thickness = similarity')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "character_network.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: character_network.png")
    plt.close()
    
    print("\n✅ Character neuron analysis complete!")
    
    # Print insights
    print("\n" + "="*60)
    print("INSIGHTS:")
    print("="*60)
    print("\n1. Character-specific neurons:")
    print("   Neurons with high variance across characters can be")
    print("   interpreted as 'hub neurons' for those entities.")
    
    print("\n2. Character similarity:")
    print("   High similarity suggests characters appear in similar")
    print("   narrative contexts (e.g., Glenarvan-Thalcave from same novel).")
    
    print("\n3. Network structure:")
    print("   Connected characters share similar neural representations,")
    print("   reflecting narrative relationships.")


if __name__ == "__main__":
    analyze_character_neurons()

