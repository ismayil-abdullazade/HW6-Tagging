# Copilot / AI agent instructions for HW-TAG

Goal: Help an AI coding assistant be immediately productive in this repository (a CS465 tagging homework).

## Quick overview
- This repo implements tagging models (HMM and CRF) in `code/` using PyTorch for tensors and saving models via torch.save (.pkl files).
- Key components:
  - `code/hmm.py` — HiddenMarkovModel (EM training, forward/backward, Viterbi, M-step).
  - `code/crf.py` — ConditionalRandomField (SGD training, conditional log-probabilities). Many methods are intentionally left as exercises.
  - `code/corpus.py` — `TaggedCorpus`, `Sentence`, integerization and OOV/BOS/EOS handling (vocab includes EOS_WORD and BOS_WORD as last two entries).
  - `code/tag.py` — CLI entrypoint. Use this for standard training/evaluation runs.
  - `code/eval.py` — evaluation helpers: `model_cross_entropy`, `viterbi_error_rate`, `write_tagging`.
  - `integerize.py` — integerizer utility (used by taggers and corpus).

## Important architecture notes (read these files together)
- Data flow: `TaggedCorpus` yields `Sentence` objects padded with BOS/EOS; models integerize these through the corpus' `integerize_sentence` before running forward/backward or Viterbi.
- HMM vs CRF:
  - HMM (in `HiddenMarkovModel`) uses EM: E_step accumulates expected counts into `self.A_counts` and `self.B_counts`; M_step converts counts to probabilities (`self.A`, `self.B`). Look at `train()` in `hmm.py` for the E/M loop and early-stopping behavior.
  - CRF (in `crf.py`) uses minibatch SGD. Training code lives in `ConditionalRandomField.train` and expects `accumulate_logprob_gradient`, `logprob_gradient_step`, `reg_gradient_step`, and `updateAB` to be implemented.
- Structural conventions: BOS_TAG and EOS_TAG must be present in tagset (see checks in `hmm.__init__`). The vocabulary stores EOS_WORD and BOS_WORD as the final two entries; many routines assume that ordering.

## Developer workflows & commands
- **Environment Setup**: Always use the `mtcourse` conda environment which has PyTorch 2.5.1 and all required dependencies:
  ```bash
  conda activate mtcourse
  ```
- Run the CLI (example — in WSL with mtcourse environment):
  - Evaluate a saved model on a dev set:
    `conda activate mtcourse && python code/tag.py data/endev -m my_model.pkl --device cpu`
  - Train an HMM (EM) and save:
    `conda activate mtcourse && python code/tag.py data/endev -t data/ensup -t data/enraw -m out.pkl`
  - Train a CRF (SGD):
    `conda activate mtcourse && python code/tag.py data/endev -t data/ensup --crf -m crf.pkl --lr 0.05 --batch_size 30 --reg 1.0`

## Homework 6 Workflow — Guiding Students Through Implementation

This section helps AI assistants guide students through the assignment step-by-step.

### Assignment Context
This is CS465 Homework 6 on Structured Prediction. Students implement HMMs (Hidden Markov Models) for POS tagging using EM training, then CRFs (Conditional Random Fields) using SGD. The assignment has these phases:

1.  **Ice cream dataset** (`icsup`, `icraw`, `icdev`) — Small toy data for debugging; results must match provided spreadsheet.
2.  **English dataset** (`ensup`, `enraw`, `endev`) — Real POS tagging task (100k training words).
3.  **Final report** — LaTeX document answering conceptual questions using empirical results.

### Key Implementation Steps (follow `code/INSTRUCTIONS.md`)
The interaction should follow these major implementation milestones, broken down into smaller, verifiable parts.

1.  **Viterbi Tagging (`hmm.py`)**: Implement `viterbi_tagging()`.
2.  **Supervised HMM Training**: Implement a simplified E-step and the `M_step()`.
3.  **Full EM for Semi-Supervised HMM**: Implement `forward_pass()` and `backward_pass()`.
4.  **CRF Implementation (`crf.py`)**: Implement `init_params`, `updateAB`, `logprob`, and the gradient/update methods.
5.  **Evaluation & Report Writing**: Run final experiments and synthesize results into the report.

### Interactive Guidance Pattern
When helping students, follow this systematic, step-by-step process:

1.  **Analyze Full Project Context**: Before starting, analyze the entire project codebase (`code/`), datasets (`data/`), the homework `homework.tex` file, and `INSTRUCTIONS.md`. This establishes a complete understanding of the goals and architecture.

2.  **Implement and Verify Incrementally**: For each function (e.g., `viterbi_tagging`):
    *   **Focus on one small part**: Address a single `NotImplementedError` or a logical block of code.
    *   **Add Debugging Probes**: Suggest temporary `print()` statements to check the state of key variables (e.g., `print(alpha_hat.shape)`, `print(f"Counts after batch: {self.A_counts}")`). This helps catch fundamental errors (like NaN values or incorrect dimensions) immediately.
    *   **Provide a Test Command**: Give the specific, complete `python code/tag.py ...` command to run the new code on a small, fast dataset (usually the `ic` dataset).
    *   **Define Expected Output**: Explain what the correct output should look like, referencing the assignment's spreadsheets or expected behavior (e.g., "The Viterbi output for icraw should be a sequence of H's and C's," or "The alpha values printed should match the ice cream spreadsheet.").

3.  **Manage Command Execution Flow**:
    *   **Execute one command at a time**: Training runs, especially on the English data, can be long.
    *   **Wait for completion**: Explicitly state: "Please run this command and paste the output. I will wait." Do not provide a new command or new code until the user confirms the previous one has finished and shares the results.

4.  **Scale Up After Verification**: Once a feature is implemented and verified on the small `ic` dataset, provide the command to run it on the full `en` dataset.

5.  **Track Results for Final Report**: Systematically collect and save every important result (accuracies, cross-entropy, answers to conceptual questions) as they are generated. Acknowledge each result with a message like: "Great. Storing this for the final report: Novel-word accuracy for the supervised HMM is X%."

### Common Gotchas for Students
-   Forgetting to handle BOS/EOS structural zeros (causes NaN in matrices).
-   Not using log-space arithmetic on English data (causes underflow).
-   Confusion between joint p(tags, words) in HMM vs conditional p(tags|words) in CRF.
-   Unigram mode: `WA` is 1×k but `A` must be k×k (broadcast the row; see `init_params`).
-   When continuing training with `--checkpoint`, vocab/tagset must match saved model (see how `tag.py` handles this).

### Deliverables Checklist
-   Modified code: `hmm.py`, `crf.py`, `tag.py`, `eval.py`, `corpus.py`, `integerize.py`
-   Trained models: `ensup_hmm.pkl`, `entrain_hmm.pkl`, `ensup_crf.pkl`
-   LaTeX report answering all questions with empirical results