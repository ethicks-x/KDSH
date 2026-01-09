import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add educational BDH repo to path
ROOT = Path(__file__).resolve().parents[2]
BDH_EDU_DIR = ROOT / "repos" / "bdh_educational"
sys.path.append(str(BDH_EDU_DIR))

from bdh import BDH, BDHParameters  # adjust if class is named differently

def main():
    print("=" * 60)
    print("BDH ARCHITECTURE INSPECTION")
    print("=" * 60)

    # ---- 1. Initialize a small BDH model ----
    vocab_size = 1000
    params = BDHParameters(
        V=vocab_size,       # vocabulary size
        T=32,               # sequence length
        H=4,                # heads
        N=256,              # neurons
        D=64,               # latent dimension
        L=2,                # layers
        dropout=0.0,        # no dropout for inspection
        use_rope=False,     # RoPE (rotary positional encoding)
        use_abs_pos=True    # absolute positional encoding
    )

    try:
        model = BDH(params)
    except TypeError as e:
        print(" Error constructing BDH with guessed config.")
        print("   Open repos/bdh_educational/bdh.py and check __init__ signature.")
        print("   Then update `config` in this script to match.")
        raise e

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model initialized with {n_params:,} parameters")
    print("\nModel:\n", model)

    print("\nNamed submodules (high level):")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")

    # ---- 2. Forward pass with dummy input ----
    batch_size = 1
    seq_len = 32

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print("\nInput shape:", x.shape)

    out = model(x)

    if isinstance(out, tuple):
        logits, state = out
    else:
        logits, state = out, None

    print("\n Forward pass OK")
    print("Logits shape:", logits.shape)
    if state is not None:
        print("State type:", type(state))
        try:
            print("State shape:", state.shape)
        except Exception:
            print("State has no .shape attribute, inspect in bdh.py")

    # ---- 3. Sparsity analysis on activations ----
    # assume logits is [B, T, D] or [B, T, vocab]
    activations = logits[0]  # [T, D] or [T, V]
    nonzero = (activations.abs() > 1e-6).float().mean().item()
    print("\nActivation sparsity:")
    print(f"  Fraction of non-zero activations: {nonzero * 100:.2f}%")

    # ---- 4. Print key methods in bdh.py for you to read ----
    print("\nSuggested things to inspect in repos/bdh_educational/bdh.py:")
    print("  - class BDH(nn.Module)")
    print("  - BDH.__init__ (graph construction, neurons)")
    print("  - BDH.forward (where σ / state is updated)")
    print("  - any Hebbian update functions (synaptic plasticity)")
    print("\nOpen that file and locate where:")
    print("  * synaptic weights are stored (σ)")
    print("  * σ is modified during forward pass (Hebbian rule)")
    print("  * sparsity is enforced (e.g., ReLU / top-k)")

    print("\nDay 1 BDH inspection script finished.")

if __name__ == '__main__':
    main()
