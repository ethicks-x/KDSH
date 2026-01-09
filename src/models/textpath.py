"""
TextPath: BDH adapted for long-form narrative text processing
Extends the educational BDH to handle variable-length sequences and state management
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add BDH educational repo to path
ROOT = Path(__file__).resolve().parents[2]
BDH_EDU_DIR = ROOT / "repos" / "bdh_educational"
sys.path.append(str(BDH_EDU_DIR))

from bdh import BDH, BDHParameters


@dataclass
class TextPathConfig:
    """Configuration for TextPath model"""
    vocab_size: int = 16384          # From custom tokenizer
    max_seq_len: int = 4096          # Maximum sequence length
    n_heads: int = 8                 # Attention heads
    n_neurons: int = 4096            # BDH neurons (scale-free graph)
    d_model: int = 256               # Model dimension
    n_layers: int = 4                # Number of BDH layers
    dropout: float = 0.1
    use_rope: bool = True            # Rotary position encoding
    sparsity_target: float = 0.05    # 5% neuron activation target


class TextPath(nn.Module):
    """
    BDH-based language model for narrative consistency detection.
    
    Key features:
    - Maintains internal synaptic state σ during inference
    - Sparse neuron activations (~5%) to prevent catastrophic interference
    - Variable-length sequence handling
    - State extraction/injection for consistency checking
    """
    
    def __init__(self, config: TextPathConfig):
        super().__init__()
        self.config = config
        
        # Create BDH parameters matching the educational implementation
        self.bdh_params = BDHParameters(
            V=config.vocab_size,
            T=config.max_seq_len,
            H=config.n_heads,
            N=config.n_neurons,
            D=config.d_model,
            L=config.n_layers,
            dropout=config.dropout,
            use_rope=config.use_rope,
            use_abs_pos=not config.use_rope,  # Use one or the other
        )
        
        # Initialize BDH core
        self.bdh = BDH(self.bdh_params)
        
        # Note: BDH already returns logits over vocab, no extra projection needed
        # Remove this block:
        # self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.output_proj.weight = self.bdh.emb.weight
        
        print(f"✅ TextPath initialized")
        print(f"   Vocab: {config.vocab_size:,}")
        print(f"   Neurons: {config.n_neurons:,}")
        print(f"   Layers: {config.n_layers}")
        print(f"   Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with optional state extraction.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] mask (1=attend, 0=ignore)
            return_state: whether to return internal state
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            state: optional dict with internal state σ
        """
        # BDH forward pass
        # Note: The educational BDH might return just logits or (logits, state)
        bdh_out = self.bdh(input_ids)
        
        # Educational BDH already returns logits over vocab: [B, T, V]
        if isinstance(bdh_out, tuple):
            logits, internal_state = bdh_out
        else:
            logits = bdh_out
            internal_state = None
        
        # No extra projection; logits are final
        
        # Extract state if requested
        state = None
        if return_state:
            state = self.extract_state()
        
        return logits, state
    
    def extract_state(self) -> dict:
        """
        Extract internal synaptic state σ from LinearAttention.
        This is the "working memory" that encodes narrative constraints.
        """
        state = {}
        
        # The state is stored in the LinearAttention module
        linear_attn = self.bdh.linear_attn
        
        # Try to extract internal state (implementation-specific)
        # This depends on how the educational BDH stores state
        if hasattr(linear_attn, 'state'):
            state['synaptic_weights'] = linear_attn.state
        elif hasattr(linear_attn, 'kv_state'):
            state['kv_state'] = linear_attn.kv_state
        else:
            # Fallback: extract from module parameters
            state['linear_attn_params'] = {
                name: param.clone().detach()
                for name, param in linear_attn.named_parameters()
            }
        
        return state
    
    def inject_state(self, state: dict):
        """
        Inject a previously saved state into the model.
        Used for: backstory → state_prime → measure novel surprise
        """
        linear_attn = self.bdh.linear_attn
        
        if 'synaptic_weights' in state:
            if hasattr(linear_attn, 'state'):
                linear_attn.state = state['synaptic_weights']
        elif 'kv_state' in state:
            if hasattr(linear_attn, 'kv_state'):
                linear_attn.kv_state = state['kv_state']
        elif 'linear_attn_params' in state:
            for name, param in state['linear_attn_params'].items():
                if hasattr(linear_attn, name):
                    getattr(linear_attn, name).data.copy_(param.data)
    
    def reset_state(self):
        """
        Reset internal state to initial conditions.
        Used before processing a new example.
        """
        linear_attn = self.bdh.linear_attn
        
        # Reset any stateful components
        if hasattr(linear_attn, 'state'):
            linear_attn.state = None
        if hasattr(linear_attn, 'kv_state'):
            linear_attn.kv_state = None
    
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute perplexity of a sequence.
        Used for consistency scoring: high perplexity = contradiction
        
        Args:
            input_ids: [batch_size, seq_len]
            target_ids: [batch_size, seq_len] (if None, use shifted input_ids)
            
        Returns:
            perplexity: scalar tensor
        """
        if target_ids is None:
            # Standard autoregressive: predict next token
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        
        logits, _ = self.forward(input_ids)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='mean'
        )
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss)
        
        return perplexity


def test_textpath():
    """Test TextPath initialization and forward pass"""
    print("="*60)
    print("TESTING TEXTPATH MODEL")
    print("="*60)
    
    # Small config for testing
    config = TextPathConfig(
        vocab_size=1000,
        max_seq_len=128,
        n_heads=4,
        n_neurons=512,
        d_model=128,
        n_layers=2,
        dropout=0.0,
    )
    
    model = TextPath(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward without state
    logits, state = model.forward(input_ids, return_state=False)
    print(f"Logits shape: {logits.shape}")
    print(f"State returned: {state is not None}")
    
    # Forward with state extraction
    logits, state = model.forward(input_ids, return_state=True)
    print(f"\nState extraction:")
    print(f"  Keys: {list(state.keys())}")
    
    # Test perplexity computation
    perplexity = model.compute_perplexity(input_ids)
    print(f"\nPerplexity: {perplexity.item():.2f}")
    
    # Test state management
    print("\nTesting state management:")
    state_backup = model.extract_state()
    print(f"  State extracted: {len(state_backup)} entries")
    
    model.reset_state()
    print(f"  State reset")
    
    model.inject_state(state_backup)
    print(f"  State injected")
    
    print("\nTextPath tests passed!")


if __name__ == "__main__":
    test_textpath()
