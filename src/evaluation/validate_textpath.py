"""
Validate TextPath learned narrative structure.
Test: Known sentences (low perplexity) vs jumbled sentences (high perplexity)
"""

import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

# Import TextPath
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from textpath import TextPath, TextPathConfig


def test_perplexity_discrimination(
    model: TextPath,
    tokenizer: Tokenizer,
    device: torch.device
):
    """
    Test if model can distinguish coherent from incoherent text
    """
    print("="*60)
    print("PERPLEXITY DISCRIMINATION TEST")
    print("="*60)
    
    # Test cases from the novels
    coherent_texts = [
        "Edmond Dantès was arrested and imprisoned in the Château d'If for fourteen years.",
        "The Count of Monte Cristo returned to Paris seeking revenge against his enemies.",
        "Lord Glenarvan organized an expedition to search for Captain Grant.",
        "Thalcave was a skilled Patagonian guide who helped the expedition.",
        "Mercedes married Fernand after Edmond was imprisoned.",
    ]
    
    incoherent_texts = [
        "Edmond Paris was imprisoned Thalcave seeking Castle Count fourteen.",
        "The Château organized Mercedes Patagonian expedition arrested enemies returned.",
        "Lord d'If was Monte Cristo guide Captain Grant imprisoned years.",
        "Thalcave married Fernand Cristo seeking Glenarvan arrested Paris revenge.",
        "Imprisoned Patagonian Count expedition organized Mercedes fourteen Castle years.",
    ]
    
    model.eval()
    
    def compute_text_perplexity(text: str) -> float:
        """Compute perplexity for a single text"""
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            perplexity = model.compute_perplexity(input_ids)
        
        return perplexity.item()
    
    print("\n COHERENT TEXT (should have LOW perplexity):")
    print("-"*60)
    coherent_ppls = []
    for text in coherent_texts:
        ppl = compute_text_perplexity(text)
        coherent_ppls.append(ppl)
        print(f"PPL: {ppl:7.2f} | {text[:60]}")
    
    print("\n INCOHERENT TEXT (should have HIGH perplexity):")
    print("-"*60)
    incoherent_ppls = []
    for text in incoherent_texts:
        ppl = compute_text_perplexity(text)
        incoherent_ppls.append(ppl)
        print(f"PPL: {ppl:7.2f} | {text[:60]}")
    
    # Statistics
    avg_coherent = sum(coherent_ppls) / len(coherent_ppls)
    avg_incoherent = sum(incoherent_ppls) / len(incoherent_ppls)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Average coherent perplexity:   {avg_coherent:.2f}")
    print(f"  Average incoherent perplexity: {avg_incoherent:.2f}")
    print(f"  Ratio (incoherent/coherent):   {avg_incoherent/avg_coherent:.2f}x")
    print("="*60)
    
    if avg_incoherent > avg_coherent * 1.5:
        print("\n Model successfully discriminates coherent from incoherent text!")
        print("   This validates that BDH learned narrative structure.")
    else:
        print("\n Model discrimination is weak.")
        print("   May need more training or larger capacity.")
    
    return avg_coherent, avg_incoherent


def test_character_knowledge(
    model: TextPath,
    tokenizer: Tokenizer,
    device: torch.device
):
    """
    Test if model learned character associations
    """
    print("\n" + "="*60)
    print("CHARACTER KNOWLEDGE TEST")
    print("="*60)
    
    # Sentences with correct character associations
    correct = [
        "Edmond Dantès was a sailor from Marseille.",
        "The Count of Monte Cristo was immensely wealthy.",
        "Lord Glenarvan owned the yacht Duncan.",
        "Abbé Faria was Dantès's mentor in prison.",
    ]
    
    # Sentences with wrong character associations
    incorrect = [
        "Edmond Dantès owned the yacht Duncan.",  # Wrong: that's Glenarvan
        "The Count of Monte Cristo was a sailor from Marseille.",  # Wrong: that's past
        "Lord Glenarvan was Dantès's mentor in prison.",  # Wrong: that's Faria
        "Abbé Faria owned the yacht Duncan.",  # Wrong: that's Glenarvan
    ]
    
    model.eval()
    
    def compute_text_perplexity(text: str) -> float:
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
        with torch.no_grad():
            perplexity = model.compute_perplexity(input_ids)
        return perplexity.item()
    
    print("\n CORRECT associations (should have LOWER perplexity):")
    print("-"*60)
    correct_ppls = []
    for text in correct:
        ppl = compute_text_perplexity(text)
        correct_ppls.append(ppl)
        print(f"PPL: {ppl:7.2f} | {text}")
    
    print("\n INCORRECT associations (should have HIGHER perplexity):")
    print("-"*60)
    incorrect_ppls = []
    for text in incorrect:
        ppl = compute_text_perplexity(text)
        incorrect_ppls.append(ppl)
        print(f"PPL: {ppl:7.2f} | {text}")
    
    avg_correct = sum(correct_ppls) / len(correct_ppls)
    avg_incorrect = sum(incorrect_ppls) / len(incorrect_ppls)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Average correct perplexity:   {avg_correct:.2f}")
    print(f"  Average incorrect perplexity: {avg_incorrect:.2f}")
    print(f"  Ratio (incorrect/correct):    {avg_incorrect/avg_correct:.2f}x")
    print("="*60)
    
    if avg_incorrect > avg_correct:
        print("\n Model shows character knowledge!")
    else:
        print("\n Model doesn't clearly distinguish character associations.")


def main():
    print("="*60)
    print("TEXTPATH VALIDATION")
    print("="*60)
    
    # Paths
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models" / "textpath_pretrained.pt"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = TextPath(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f" Model loaded (params: {sum(p.numel() for p in model.parameters()):,})")
    
    # Run tests
    test_perplexity_discrimination(model, tokenizer, device)
    test_character_knowledge(model, tokenizer, device)
    
    print("\n Validation complete!")


if __name__ == "__main__":
    main()