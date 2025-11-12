# HW-TAG: HMM and CRF Taggers for Part-of-Speech Tagging

## Overview
This project implements Hidden Markov Model (HMM) and Conditional Random Field (CRF) taggers for part-of-speech tagging, as part of CS465 Natural Language Processing at Johns Hopkins University.

## Implementation Status
✅ **Complete**: All core algorithms implemented and tested
- HMM with Viterbi decoding
- HMM training via EM (Expectation-Maximization)
- CRF with discriminative training via SGD
- Full log-space arithmetic for numerical stability

## Quick Start

### Environment Setup
```bash
conda activate mtcourse
```

### Testing on Ice Cream Data
```bash
cd code
python test_ic.py  # Tests HMM and CRF on small ice cream dataset
```

### Training and Evaluation

#### Supervised HMM
```bash
python code/tag.py data/endev --model ensup_hmm.pkl --train data/ensup
```

#### Semi-supervised HMM
```bash
python code/tag.py data/endev --model entrain_hmm.pkl --train data/ensup data/enraw
```

#### Supervised CRF
```bash
python code/tag.py data/endev --model ensup_crf.pkl --crf --train data/ensup --lr 0.05 --batch_size 30 --reg 1.0
```

#### Evaluate Existing Model
```bash
python code/tag.py data/endev --model my_model.pkl
```

## Command-Line Options

### Model Type
- `--crf`: Train a CRF instead of HMM
- `--unigram`: Use unigram model (baseline, ignores context)

### Training
- `-t, --train FILE [FILE ...]`: Training data files
- `-c, --checkpoint FILE`: Continue from saved model
- `--tolerance FLOAT`: Convergence tolerance (default: 0.001)
- `--max_steps INT`: Maximum training sentences (default: 50000)

### HMM-Specific
- `-λ, --lambda FLOAT`: Add-λ smoothing for M-step (default: 0)

### CRF-Specific
- `--lr FLOAT`: Learning rate (default: 0.05)
- `--batch_size INT`: Minibatch size (default: 30)
- `--reg FLOAT`: L2 regularization coefficient (default: 0)
- `--eval_interval INT`: Evaluate every N sentences (default: 2000)

### Evaluation
- `--loss {cross_entropy,viterbi_error}`: Loss function for evaluation
- `-o, --output_file FILE`: Where to save tagging output
- `-e, --eval_file FILE`: Where to log evaluation messages

## File Structure

```
code/
├── hmm.py              # HMM implementation (EM training, Viterbi, forward-backward)
├── crf.py              # CRF implementation (SGD training, conditional probabilities)
├── tag.py              # Command-line interface
├── corpus.py           # Corpus management and integerization
├── eval.py             # Evaluation metrics (accuracy, cross-entropy)
├── integerize.py       # Utility for mapping objects to integers
├── test_ic.py          # Tests on ice cream data
└── test_en.py          # Tests on English data

data/
├── icsup, icraw, icdev  # Ice cream training/test data (tiny)
├── ensup, enraw, endev  # English training/test data (~100k tokens)
└── ensup-tiny           # Small English subset for quick testing
```

## Implementation Details

### HMM (`hmm.py`)

#### Viterbi Algorithm
- Finds the most probable tag sequence for a sentence
- Uses dynamic programming with backpointers
- Time complexity: O(nk²) where n = sentence length, k = number of tags

#### Forward-Backward Algorithm
- **Forward pass**: Computes α probabilities (probability of prefix ending in each tag)
- **Backward pass**: Computes β probabilities (probability of suffix starting from each tag)
- **E-step**: Accumulates expected transition and emission counts
- **M-step**: Normalizes counts into probabilities with add-λ smoothing

#### Numerical Stability
- All computations in log-space to avoid underflow
- Uses `torch.logsumexp()` for stable probability summation
- Thresholding: only accumulate counts if log_posterior > -100

#### Structural Zeros
- No transitions TO BOS_TAG
- No transitions FROM EOS_TAG  
- No emissions from BOS_TAG or EOS_TAG

### CRF (`crf.py`)

#### Parametrization
- Weight matrices: `WA` (transitions), `WB` (emissions)
- Potentials: `A = exp(WA)`, `B = exp(WB)`
- Inherits forward-backward from HMM (with potentials instead of probabilities)

#### Training
- Discriminative: maximizes conditional log-likelihood p(tags | words)
- Gradient: observed counts - expected counts
- SGD with minibatch updates
- L2 regularization via weight decay

#### Conditional Probability
```
log p(tags | words) = log p(tags, words) - log p(words)
                    = log Z_supervised - log Z_unsupervised
```

### Corpus Management (`corpus.py`)
- Handles tagged and untagged sentences
- Integerizes words and tags for tensor operations
- Manages vocabulary and tagset
- Special tokens: BOS_WORD, EOS_WORD, BOS_TAG, EOS_TAG, OOV_WORD

### Evaluation (`eval.py`)
- **Cross-entropy**: Measures model uncertainty (lower is better)
- **Tagging accuracy**: Percentage of correctly tagged tokens
- Breaks down by word type: known/seen/novel words

## Dataset Format

### Supervised Data (e.g., `ensup`)
```
Papa/N ate/V the/D caviar/N with/P a/D spoon/N ./.
```

### Unsupervised Data (e.g., `enraw`)
```
Papa ate the caviar with a spoon .
```

## Expected Results

### Ice Cream Data
- Small toy dataset for debugging
- Should match provided spreadsheet values exactly
- Viterbi path: sequence of H's (hot) and C's (cold)
- EM converges in ~10 iterations

### English Data
- ~100k tokens in training data
- ~1k tokens in dev data
- Expected accuracy: 90-95% (with reduced tagset)
- Training time: ~5-10 minutes for HMM, ~10-20 minutes for CRF

### Key Findings (Expected)
1. **Bigram > Unigram**: Context helps significantly
2. **Semi-supervised**: May help or hurt depending on domain match
3. **CRF ≈ HMM** (with simple features): Similar accuracy
4. **Novel words**: Hardest to tag correctly (60-70% accuracy)

## Deliverables

### Code (Submit)
- `hmm.py`, `crf.py`, `tag.py`, `eval.py`, `corpus.py`, `integerize.py`

### Models (Generate & Submit)
- `ensup_hmm.pkl`: Supervised HMM
- `entrain_hmm.pkl`: Semi-supervised HMM
- `ensup_crf.pkl`: Supervised CRF

### Report
- `hw-tag.tex`: Completed with all question answers

## Troubleshooting

### NaN values during training
- **Cause**: Numerical underflow in forward-backward
- **Fix**: Ensure log-space arithmetic is used throughout
- **Check**: Look for `torch.log()` calls without log-space computation

### Assertion Error: "expected transition counts from EOS are not zero"
- **Cause**: Backward pass incorrectly accumulates counts from EOS
- **Fix**: Skip s == EOS_TAG in transition count accumulation

### Slow training
- **Cause**: Too many small operations, not using tensor operations effectively
- **Fix**: Precompute log(A) and log(B) once per sentence
- **Tip**: Progress bar should show ~30-50 sentences/sec for HMM

### Model doesn't converge
- **Cause**: Learning rate too high (CRF) or tolerance too strict
- **Fix**: Try --lr 0.01 (instead of 0.05) or --tolerance 0.01

## References
- Reading handout: Comprehensive guide to HMMs and CRFs
- Ice cream spreadsheet: Step-by-step algorithm verification
- CS465 course materials: https://www.cs.jhu.edu/~jason/465/

## Authors
Implementation for CS465 Fall 2025, Johns Hopkins University
