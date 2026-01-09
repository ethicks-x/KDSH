"""
Analyze if model learned geographic constraints (specific to Castaways novel).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from textpath import TextPath


def analyze_geographic_neurons():
    """
    Test if model learned geographic/numerical concepts.
    """
    print("="*60)
    print("GEOGRAPHIC NEURON ANALYSIS")
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
    
    # Test sentences with geographic terms
    geo_sentences = {
        'Coordinates': [
            "The ship sailed along the 37th parallel",
            "They searched at latitude 37 degrees",
            "The coordinates pointed to the south"
        ],
        'Mountains': [
            "They crossed the Andes mountains",
            "The peaks of the Andes were visible",
            "High in the mountains they found"
        ],
        'Ocean': [
            "The Pacific Ocean stretched endlessly",
            "They sailed across the ocean",
            "The vast ocean surrounded them"
        ],
        'Patagonia': [
            "The plains of Patagonia were endless",
            "In Patagonia they met Thalcave",
            "The Patagonian landscape was harsh"
        ]
    }
    
    print("\nTesting perplexity on geographic facts...")
    
    # Test correct vs incorrect geographic statements
    correct_facts = [
        "The expedition searched along the 37th parallel.",
        "Thalcave was a Patagonian guide.",
        "They crossed the Andes mountains.",
        "Lord Glenarvan sailed on the Duncan."
    ]
    
    incorrect_facts = [
        "The expedition searched along the 90th parallel.",  # Invalid latitude
        "Thalcave was a European guide.",  # Wrong origin
        "They crossed the Himalayan mountains.",  # Wrong mountains
        "Lord Glenarvan sailed on the Pharaon."  # Wrong ship (from Monte Cristo)
    ]
    
    def compute_perplexity(text):
        ids = torch.tensor([tokenizer.encode(text).ids]).to(device)
        with torch.no_grad():
            ppl = model.compute_perplexity(ids)
        return ppl.item()
    
    print("\n📊 CORRECT geographic facts (should have LOWER perplexity):")
    correct_ppls = []
    for fact in correct_facts:
        ppl = compute_perplexity(fact)
        correct_ppls.append(ppl)
        print(f"  PPL: {ppl:7.2f} | {fact}")
    
    print("\n📊 INCORRECT geographic facts (should have HIGHER perplexity):")
    incorrect_ppls = []
    for fact in incorrect_facts:
        ppl = compute_perplexity(fact)
        incorrect_ppls.append(ppl)
        print(f"  PPL: {ppl:7.2f} | {fact}")
    
    avg_correct = np.mean(correct_ppls)
    avg_incorrect = np.mean(incorrect_ppls)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Average correct PPL:   {avg_correct:.2f}")
    print(f"  Average incorrect PPL: {avg_incorrect:.2f}")
    print(f"  Ratio:                 {avg_incorrect/avg_correct:.2f}x")
    print("="*60)
    
    if avg_incorrect > avg_correct * 1.3:
        print("\n✅ Model shows geographic constraint knowledge!")
    else:
        print("\n⚠️ Weak geographic constraint learning.")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(correct_facts))
    width = 0.35
    
    ax.bar(x - width/2, correct_ppls, width, label='Correct Facts', color='green', alpha=0.7)
    ax.bar(x + width/2, incorrect_ppls, width, label='Incorrect Facts', color='red', alpha=0.7)
    
    ax.set_xlabel('Statement Index')
    ax.set_ylabel('Perplexity')
    ax.set_title('Geographic Constraint Validation\n(Lower PPL = Model believes it more)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in range(len(correct_facts))])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "geographic_constraint_validation.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: geographic_constraint_validation.png")
    plt.close()
    
    print("\n✅ Geographic analysis complete!")


if __name__ == "__main__":
    analyze_geographic_neurons()

