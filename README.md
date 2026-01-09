# KDSH

Repository for the KDSH project.

# KDSH 2026 Track B - Task Checklist

## DAY 1: Infrastructure & Data Pipeline ✅ COMPLETE

### Environment Setup
- [x] Create conda environment (kds)
- [x] Install PyTorch, Pathway, tokenizers
- [x] Clone pathwaycom/bdh (official)
- [x] Clone krychu/bdh (educational)
- [x] Run boardpath.py successfully
- [x] Verify CUDA/GPU setup

### Data Engineering
- [x] Create project directory structure
- [x] Download train.csv, test.csv, novels
- [x] Build Pathway ingestion pipeline (ingest.py)
- [x] Clean Gutenberg headers/footers
- [x] Map string labels to integers (consistent=1, contradict=0)
- [x] Handle NaN values in caption field

### Tokenizer Training
- [x] Create train_tokenizer.py
- [x] Train BPE tokenizer (16K vocab) on both novels
- [x] Validate character names tokenization (1-2 tokens)
- [x] Verify compression ratio (3-4 chars/token)
- [x] Save tokenizer to models/custom_tokenizer.json

### BDH Architecture Understanding
- [x] Create bdh_inspect.py
- [x] Initialize BDH model (179K params)
- [x] Run forward pass successfully
- [x] Identify LinearAttention as core component
- [x] Confirm logits output shape [B, T, V]
- [x] Note: state σ maintained internally

---

## DAY 2: BDH Adaptation for Text ⏳ PENDING

### Morning: TextPath Model Creation (09:00-13:00)
- [ ] Create src/models/textpath.py
- [ ] Modify BDH to accept variable-length sequences
- [ ] Implement causal masking for autoregressive prediction
- [ ] Add state extraction/injection methods
- [ ] Test forward pass on tokenized text
- [ ] Verify internal neuron sparsity (~5%)

### Afternoon: Unsupervised Pre-training (13:00-18:00)
- [ ] Create training script (pretrain_textpath.py)
- [ ] Load novels via Pathway
- [ ] Tokenize with custom tokenizer
- [ ] Implement next-token prediction loss
- [ ] Train on The Count of Monte Cristo (Epoch 1)
- [ ] Train on In Search of the Castaways (Epoch 1)
- [ ] Save checkpoint: models/textpath_pretrained.pt
- [ ] Monitor: loss, sparsity, memory usage

### Evening: Validation & State Testing (18:00-20:00)
- [ ] Test perplexity on known sentences (should be low)
- [ ] Test perplexity on jumbled sentences (should be high)
- [ ] Verify state persistence across 4096+ tokens
- [ ] Test state reset functionality
- [ ] Document findings in docs/day2_validation.md

---

## DAY 3: Consistency Classifier Logic ⏳ PENDING

### Morning: State Management System (09:00-12:00)
- [ ] Create src/models/state_manager.py
- [ ] Implement state save/load from LinearAttention
- [ ] Build "State Carry-Over" data loader
- [ ] Test: backstory → state_prime → novel
- [ ] Ensure no state reset between stages

### Afternoon: RAG Integration (12:00-17:00)
- [ ] Set up Pathway Vector Store for novels
- [ ] Index both novels with embeddings
- [ ] Implement retrieval: given backstory → top-K novel chunks
- [ ] Test retrieval quality on train examples
- [ ] Create retrieval_config.json (K=5 initial)

### Evening: Perplexity Delta Classifier (17:00-20:00)
- [ ] Create src/evaluation/consistency_scorer.py
- [ ] Implement: score = avg_loss(novel_chunks | state_primed)
- [ ] Run on all train.csv examples
- [ ] Extract features: [loss, perplexity, state_norm]
- [ ] Train logistic regression: features → label (0/1)
- [ ] Evaluate on train set (accuracy, F1)
- [ ] Save classifier: models/consistency_classifier.pkl

---

## DAY 4: Optimization & Visualization ⏳ PENDING

### Morning: Hyperparameter Tuning (09:00-13:00)
- [ ] Grid search: n_neurons [1024, 4096, 8192]
- [ ] Grid search: retrieval_k [3, 5, 10]
- [ ] Grid search: Hebbian learning rate η
- [ ] Test: chronological vs. context-based retrieval
- [ ] Select best config based on train accuracy

### Afternoon: Interpretability Artifacts (13:00-17:00)
- [ ] Create visualization script (visualize_consistency.py)
- [ ] Generate synaptic heatmap: σ_base vs σ_primed
- [ ] Compare: consistent backstory vs contradictory
- [ ] Visualize hub neurons (character neurons)
- [ ] Create geographic neuron analysis (Castaways)
- [ ] Save all plots to visualizations/

### Evening: Castaways-Specific Testing (17:00-20:00)
- [ ] Analyze geographic constraint examples
- [ ] Test date/number tokenization handling
- [ ] Identify failure modes (negation, absence)
- [ ] Create failure_analysis.md
- [ ] Adjust preprocessing if needed

---

## DAY 5: Final Inference & Submission ⏳ PENDING

### Morning: Test Set Inference (09:00-12:00)
- [ ] Create inference.py script
- [ ] Load best model + classifier from Day 4
- [ ] Run full pipeline on test.csv
- [ ] Generate results.csv (id, prediction, rationale)
- [ ] Validate output format
- [ ] Sanity check: predictions distribution

### Afternoon: Report Writing (12:00-16:00)
- [ ] Write 10-page report (report.pdf)
  - [ ] Section 1: Approach (BDH as Dynamic State Estimator)
  - [ ] Section 2: Long Context Handling (Hebbian Plasticity)
  - [ ] Section 3: Causal Signal Detection (Perplexity Delta)
  - [ ] Section 4: Results (train accuracy, visualizations)
  - [ ] Section 5: Limitations (negation, inference speed)
- [ ] Include all visualizations from Day 4
- [ ] Add architecture diagrams
- [ ] Proofread and format

### Evening: Packaging & Submission (16:00-18:00)
- [ ] Create requirements.txt
- [ ] Write README.md with reproduction steps
- [ ] Create reproduce_results.py
- [ ] Test on clean environment
- [ ] Zip: <TEAMNAME>_KDSH_2026.zip
- [ ] Verify zip contains: code/, models/, results.csv, report.pdf
- [ ] Submit before deadline

---

## CRITICAL PATH ITEMS 🔥

- [ ] Day 2: State extraction from LinearAttention (blocker for Day 3)
- [ ] Day 3: RAG retrieval quality (affects accuracy)
- [ ] Day 4: Synaptic visualizations (required for report)
- [ ] Day 5: Report completion (50% of evaluation)

## RISK MITIGATION

- **Memory OOM**: Use chunking, reduce n_neurons if needed
- **Low train accuracy**: Fall back to simpler RAG + LLM baseline
- **Time pressure Day 5**: Start report outline on Day 4 evening
- **Visualization bugs**: Have backup static analysis ready

---

**Current Status**: Day 1 complete, ready for Day 2
**Next Action**: Implement textpath.py (BDH text adaptation)