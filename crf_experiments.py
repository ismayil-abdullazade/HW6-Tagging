#!/usr/bin/env python3
"""
Comprehensive CRF vs HMM experiments for Question 4.
Tests different hyperparameters, unigram mode, and metrics.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(name, train_args, eval_args, description):
    """Run a training and evaluation experiment."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {description}")
    print('='*80)
    
    # Training
    train_cmd = ["python", "code/tag.py"] + train_args
    print(f"\n▶ Training: {' '.join(train_cmd)}")
    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if train_result.returncode != 0:
        print(f"❌ Training failed:")
        print(train_result.stderr)
        return False
    
    # Show training summary
    for line in train_result.stdout.split('\n'):
        if 'Total training time' in line or 'Cross-entropy' in line:
            print(f"  {line.strip()}")
    
    # Evaluation
    eval_cmd = ["python", "code/tag.py"] + eval_args
    print(f"\n▶ Evaluating: {' '.join(eval_cmd)}")
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    if eval_result.returncode != 0:
        print(f"❌ Evaluation failed:")
        print(eval_result.stderr)
        return False
    
    # Show evaluation results
    print("\n📊 Results:")
    for line in eval_result.stdout.split('\n'):
        if 'accuracy:' in line or 'Cross-entropy:' in line:
            print(f"  {line.strip()}")
    
    return True

def main():
    print("="*80)
    print("CRF VS HMM COMPREHENSIVE EXPERIMENTS")
    print("="*80)
    
    experiments = []
    
    # ===== 4(a): CRF vs HMM on supervised data (ensup) =====
    
    # Experiment 1: HMM Supervised Baseline (already done, just eval)
    experiments.append({
        "name": "1. HMM Supervised Baseline",
        "train_args": None,  # Already trained
        "eval_args": ["data/endev", "-m", "en_hmm.pkl", "-o", "exp1_hmm_sup.output"],
        "description": "Baseline HMM trained on ensup only"
    })
    
    # Experiment 2: CRF Supervised Baseline (already done, just eval)
    experiments.append({
        "name": "2. CRF Supervised Baseline",
        "train_args": None,  # Already trained
        "eval_args": ["data/endev", "-m", "en_crf.pkl", "-o", "exp2_crf_sup.output"],
        "description": "Baseline CRF trained on ensup only"
    })
    
    # Experiment 3: CRF with different learning rate
    experiments.append({
        "name": "3. CRF Lower Learning Rate",
        "train_args": [
            "data/endev", "-t", "data/ensup", "--crf", 
            "-m", "exp3_crf_lr001.pkl", 
            "--lr", "0.01", "--reg", "1.0", "--batch_size", "30"
        ],
        "eval_args": ["data/endev", "-m", "exp3_crf_lr001.pkl", "-o", "exp3_crf_lr001.output"],
        "description": "CRF with lr=0.01 (vs default 0.05)"
    })
    
    # Experiment 4: CRF with higher regularization
    experiments.append({
        "name": "4. CRF Higher Regularization",
        "train_args": [
            "data/endev", "-t", "data/ensup", "--crf",
            "-m", "exp4_crf_reg5.pkl",
            "--lr", "0.05", "--reg", "5.0", "--batch_size", "30"
        ],
        "eval_args": ["data/endev", "-m", "exp4_crf_reg5.pkl", "-o", "exp4_crf_reg5.output"],
        "description": "CRF with reg=5.0 (vs default 1.0) - stronger regularization"
    })
    
    # Experiment 5: CRF Unigram Mode
    experiments.append({
        "name": "5. CRF Unigram Mode",
        "train_args": [
            "data/endev", "-t", "data/ensup", "--crf", "--unigram",
            "-m", "exp5_crf_unigram.pkl",
            "--lr", "0.05", "--reg", "1.0", "--batch_size", "30"
        ],
        "eval_args": ["data/endev", "-m", "exp5_crf_unigram.pkl", "-o", "exp5_crf_unigram.output"],
        "description": "CRF with unigram mode (no transition features)"
    })
    
    # Experiment 6: HMM Unigram Mode (for comparison)
    experiments.append({
        "name": "6. HMM Unigram Mode",
        "train_args": [
            "data/endev", "-t", "data/ensup", "--unigram",
            "-m", "exp6_hmm_unigram.pkl"
        ],
        "eval_args": ["data/endev", "-m", "exp6_hmm_unigram.pkl", "-o", "exp6_hmm_unigram.output"],
        "description": "HMM with unigram mode (uniform transition)"
    })
    
    # ===== 4(b): Adding unsupervised data (enraw) =====
    
    # Experiment 7: HMM Semi-supervised (already done, just eval)
    experiments.append({
        "name": "7. HMM Semi-supervised",
        "train_args": None,  # Already trained
        "eval_args": ["data/endev", "-m", "en_hmm_raw.pkl", "-o", "exp7_hmm_raw.output"],
        "description": "HMM trained on ensup + enraw"
    })
    
    # Experiment 8: CRF Semi-supervised (already done, just eval)
    experiments.append({
        "name": "8. CRF Semi-supervised",
        "train_args": None,  # Already trained
        "eval_args": ["data/endev", "-m", "en_crf_raw.pkl", "-o", "exp8_crf_raw.output"],
        "description": "CRF trained on ensup + enraw"
    })
    
    # Experiment 9: CRF with larger batch size on semi-supervised
    experiments.append({
        "name": "9. CRF Semi-supervised Large Batch",
        "train_args": [
            "data/endev", "-t", "data/ensup", "-t", "data/enraw", "--crf",
            "-m", "exp9_crf_raw_batch100.pkl",
            "--lr", "0.05", "--reg", "1.0", "--batch_size", "100"
        ],
        "eval_args": ["data/endev", "-m", "exp9_crf_raw_batch100.pkl", "-o", "exp9_crf_raw_batch100.output"],
        "description": "CRF with batch_size=100 on ensup+enraw"
    })
    
    # Run experiments
    results = []
    for exp in experiments:
        try:
            if exp["train_args"] is not None:
                success = run_experiment(
                    exp["name"],
                    exp["train_args"],
                    exp["eval_args"],
                    exp["description"]
                )
            else:
                # Just evaluation
                print(f"\n{'='*80}")
                print(f"EXPERIMENT: {exp['name']}")
                print(f"Description: {exp['description']}")
                print('='*80)
                
                eval_cmd = ["python", "code/tag.py"] + exp["eval_args"]
                print(f"\n▶ Evaluating: {' '.join(eval_cmd)}")
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
                
                if eval_result.returncode != 0:
                    print(f"❌ Evaluation failed:")
                    print(eval_result.stderr)
                    success = False
                else:
                    print("\n📊 Results:")
                    for line in eval_result.stdout.split('\n'):
                        if 'accuracy:' in line or 'Cross-entropy:' in line:
                            print(f"  {line.strip()}")
                    success = True
            
            results.append((exp["name"], success))
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append((exp["name"], False))
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print('='*80)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print("\n📁 Output files saved with prefix 'exp*'")
    print("📊 Use these results to answer Question 4(a) and 4(b)")

if __name__ == '__main__':
    main()
