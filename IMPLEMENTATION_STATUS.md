# HW-TAG Implementation Status

## ✅ Completed Implementation

### HMM (Hidden Markov Model) - `code/hmm.py`
- **Viterbi algorithm** (`viterbi_tagging`): Finds most probable tag sequence using dynamic programming with backpointers
- **M-step** (`M_step`): Normalizes expected counts into probabilities with add-λ smoothing, handles both unigram and bigram models
- **Forward algorithm** (`forward_pass`): Computes marginal probability p(words) using log-space arithmetic for numerical stability
- **Backward algorithm** (`backward_pass`): Computes beta values and accumulates expected transition/emission counts for EM training
- **Numerical stability**: All algorithms use log-space computation to avoid underflow on real data

### CRF (Conditional Random Field) - `code/crf.py`
- **Parameter initialization** (`init_params`): Random initialization of weight matrices WA and WB with structural zeros
- **Potential computation** (`updateAB`): Converts weights to potentials using exp() for both unigram and bigram models
- **Conditional probability** (`logprob`): Computes log p(tags|words) = log p(tags, words) - log p(words)
- **Gradient accumulation** (`accumulate_logprob_gradient`): Computes gradient as observed counts minus expected counts
- **Gradient steps** (`logprob_gradient_step`, `reg_gradient_step`): SGD updates with L2 regularization (weight decay)

### Verification
- ✅ Ice cream dataset: All algorithms match spreadsheet values (Viterbi path, forward probabilities, EM iterations)
- ✅ HMM training converges properly on supervised data (icsup)
- ✅ CRF training works on supervised data (icsup)
- ✅ Numerical stability confirmed on small English subset (ensup-tiny)

## 🔄 In Progress

### English Data Training
- **Currently running**: Supervised HMM training on `data/ensup` (100k tokens)
- Background terminal ID: `0cbd036c-6849-4d59-ae76-26843aed7c9a`
- Output will be saved to: `ensup_hmm.pkl`

## 📋 TODO

### Training Tasks
1. Complete supervised HMM training on ensup
2. Train semi-supervised HMM on ensup + enraw
3. Train supervised CRF on ensup
4. Compare accuracies and cross-entropies

### Homework Questions (from hw-tag.tex)
Need to answer based on empirical results:
- Question 2(a): Why initialize α_BOS(0)=1 and β_EOS(n+1)=1?
- Question 2(b): Why is perplexity lower on raw vs dev?
- Question 2(c): Why not include dev vocab?
- Question 2(d): Did semi-supervised training help?
- Question 2(e): Why might semi-supervised approach help?
- Question 2(f): Why might it not always help?
- Question 2(g): Bigram HMM vs unigram HMM comparison
- Question 4(a): CRF vs HMM comparison

## 📁 Deliverables

### Code Files (to submit)
- ✅ `hmm.py` - Complete with all methods implemented
- ✅ `crf.py` - Complete with all methods implemented
- ✅ `tag.py` - Command-line interface (unchanged from starter)
- ✅ `eval.py` - Evaluation functions (unchanged from starter)
- ✅ `corpus.py` - Corpus management (unchanged from starter)
- ✅ `integerize.py` - Integerizer utility (unchanged from starter)

### Model Files (to generate)
- ⏳ `ensup_hmm.pkl` - Currently training
- ⏳ `entrain_hmm.pkl` - Semi-supervised HMM (to train)
- ⏳ `ensup_crf.pkl` - Supervised CRF (to train)

### Report
- ⏳ Completed `hw-tag.tex` with answers to all questions

## 🎯 Key Implementation Details

### Numerical Stability Strategy
- All probabilities stored and computed in log-space
- Use `torch.logsumexp()` for stable summation of probabilities
- Precompute log(A) and log(B) matrices once per forward/backward pass
- Threshold for very small probabilities: only add counts if log_posterior > -100

### Structural Zeros
Properly handled throughout:
- No transitions TO BOS_TAG (A[:, bos_t] = 0)
- No transitions FROM EOS_TAG (A[eos_t, :] = 0)
- No emissions from BOS_TAG or EOS_TAG (B[bos_t,:] = 0, B[eos_t,:] = 0)

### Unigram Model Support
- WA is 1×k for unigram (single row), but A is expanded to k×k for compatibility
- Transition counts summed over all previous states for gradient updates
