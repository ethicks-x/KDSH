"""
Pre-train TextPath on The Count of Monte Cristo and In Search of the Castaways.
Goal: Learn the base narrative state of each novel.
"""

import sys
from pathlib import Path
from typing import List
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

# Import TextPath
from textpath import TextPath, TextPathConfig


class NovelDataset(Dataset):
    """Dataset for chunked novel text"""
    
    def __init__(
        self,
        novel_path: Path,
        tokenizer: Tokenizer,
        chunk_size: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Load and tokenize full novel
        print(f"Loading {novel_path.name}...")
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean Gutenberg headers (basic version)
        text = self._clean_gutenberg(text)
        
        print(f"Tokenizing {len(text):,} characters...")
        encoding = tokenizer.encode(text)
        self.token_ids = encoding.ids
        
        print(f"  → {len(self.token_ids):,} tokens")
        
        # Create overlapping chunks
        self.chunks = []
        for i in range(0, len(self.token_ids) - chunk_size, stride):
            chunk = self.token_ids[i:i + chunk_size]
            if len(chunk) == chunk_size:
                self.chunks.append(chunk)
        
        print(f"  → {len(self.chunks):,} chunks (size={chunk_size}, stride={stride})")
    
    def _clean_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg headers/footers"""
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
        ]
        
        start_idx = 0
        for marker in start_markers:
            if marker in text:
                start_idx = text.find(marker)
                start_idx = text.find('\n', start_idx) + 1
                break
        
        end_idx = len(text)
        for marker in end_markers:
            if marker in text:
                end_idx = text.find(marker)
                break
        
        return text[start_idx:end_idx].strip()
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return torch.tensor(chunk, dtype=torch.long)


def train_epoch(
    model: TextPath,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Autoregressive targets: predict next token
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            target_ids.reshape(-1),
            reduction='mean'
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        batch_loss = loss.item()
        batch_tokens = target_ids.numel()
        total_loss += batch_loss * batch_tokens
        total_tokens += batch_tokens
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'ppl': f'{torch.exp(loss).item():.2f}',
        })
    
    avg_loss = total_loss / total_tokens
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'perplexity': avg_ppl,
    }


def main():
    print("="*60)
    print("TEXTPATH PRE-TRAINING ON NOVELS")
    print("="*60)
    
    # Paths
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = ROOT / "Dataset" / "Books"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    output_dir = ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocabulary size: {vocab_size:,}")
    
    # Create model
    print("\nInitializing TextPath model...")
    config = TextPathConfig(
        vocab_size=vocab_size,
        max_seq_len=512,      # Start with 512 for faster training
        n_heads=8,
        n_neurons=2048,       # Medium size for M2 Mac
        d_model=256,
        n_layers=4,
        dropout=0.1,
        use_rope=True,
    )
    
    model = TextPath(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Load datasets
    novels = [
        ("The Count of Monte Cristo", data_dir / "The Count of Monte Cristo.txt"),
        ("In search of the castaways", data_dir / "In search of the castaways.txt"),
    ]
    
    training_config = {
        'chunk_size': 512,
        'stride': 256,
        'batch_size': 4,  # Small batch for M2 Mac
        'epochs': 2,
        'model_config': config.__dict__,
    }
    
    print("\nTraining configuration:")
    print(json.dumps(training_config, indent=2))
    
    # Train on each novel
    all_metrics = []
    
    for novel_name, novel_path in novels:
        print("\n" + "="*60)
        print(f"TRAINING ON: {novel_name}")
        print("="*60)
        
        # Create dataset
        dataset = NovelDataset(
            novel_path=novel_path,
            tokenizer=tokenizer,
            chunk_size=training_config['chunk_size'],
            stride=training_config['stride'],
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 for M2 Mac compatibility
        )
        
        # Train epochs
        for epoch in range(1, training_config['epochs'] + 1):
            metrics = train_epoch(model, dataloader, optimizer, device, epoch)
            
            print(f"\nEpoch {epoch} completed:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
            
            all_metrics.append({
                'novel': novel_name,
                'epoch': epoch,
                **metrics
            })
        
        # Save checkpoint after each novel
        checkpoint_path = output_dir / f"textpath_{novel_name.replace(' ', '_').lower()}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'metrics': all_metrics,
        }, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "textpath_pretrained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_config': training_config,
        'metrics': all_metrics,
    }, final_path)
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE")
    print(f"Final model saved: {final_path}")
    print("="*60)
    
    # Print summary
    print("\nTraining Summary:")
    for metric in all_metrics:
        print(f"  {metric['novel']} Epoch {metric['epoch']}: "
              f"Loss={metric['loss']:.4f}, PPL={metric['perplexity']:.2f}")


if __name__ == "__main__":
    main()