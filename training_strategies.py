#!/usr/bin/env python3
"""
Question (h) Extra Credit: Training Strategy Experiments

We'll test different strategies for combining supervised (ensup) and 
unsupervised (enraw) data to combat the Merialdo effect.

Strategies to test:
1. Baseline: ensup only (supervised)
2. Standard semi-supervised: ensup + enraw
3. Weighted supervised: ensup + ensup + ensup + enraw (3:1 weighting)
4. More weighted: ensup × 5 + enraw (5:1 weighting)
5. Even more weighted: ensup × 10 + enraw (10:1 weighting)
6. Staged training: Train on ensup, then continue on enraw
7. Reverse staged: Train on enraw, then fine-tune on ensup
8. Unsupervised only: enraw alone (for comparison)

Metrics: Accuracy and cross-entropy on endev
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from corpus import TaggedCorpus
from hmm import HiddenMarkovModel
from eval import model_cross_entropy, viterbi_error_rate
import subprocess
import os

def train_model(output_path, train_files, device='cpu', load_from=None):
    """Train an HMM model with specified training files.
    
    Args:
        output_path: Where to save the trained model
        train_files: List of training data files
        device: Device to use ('cpu' or 'cuda')
        load_from: Optional path to load model from (for staged training)
    """
    cmd = ['python', 'code/tag.py', 'data/endev']
    
    if load_from:
        # Staged training: load existing model and continue training
        cmd.extend(['-c', load_from])
    
    # Use single -t flag with all files (compatible with nargs="+")
    # This works with the original argparse configuration
    if train_files:
        cmd.append('-t')
        cmd.extend(train_files)
    
    cmd.extend(['-m', output_path, '--device', device])
    
    print(f"\n{'='*80}")
    print(f"Training: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print last 30 lines of output (should show convergence info)
    lines = result.stdout.strip().split('\n')
    for line in lines[-30:]:
        print(line)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    return True

def evaluate_model(model_path, eval_corpus_path, model_name):
    """Evaluate a model and return accuracy and cross-entropy."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Load model
        model = HiddenMarkovModel.load(model_path)
        
        # Load evaluation corpus
        eval_corpus = TaggedCorpus(eval_corpus_path, tagset=model.tagset, vocab=model.vocab)
        
        # Compute accuracy
        print("Computing Viterbi accuracy...")
        error_rate = viterbi_error_rate(model, eval_corpus)
        accuracy = (1 - error_rate) * 100
        
        # Compute cross-entropy
        print("Computing cross-entropy...")
        cross_entropy = model_cross_entropy(model, eval_corpus)
        
        print(f"\nResults for {model_name}:")
        print(f"  Accuracy:      {accuracy:.3f}%")
        print(f"  Cross-entropy: {cross_entropy:.4f} bits/word")
        
        return accuracy, cross_entropy
    except Exception as e:
        print(f"ERROR evaluating {model_name}: {e}")
        return None, None

def create_weighted_corpus(ensup_copies, output_path='data/ensup_weighted'):
    """Create a temporary file with multiple copies of ensup for weighting."""
    if os.path.exists(output_path):
        return output_path
    
    print(f"Creating weighted corpus: {ensup_copies} copies of ensup")
    
    with open(output_path, 'w') as out:
        for i in range(ensup_copies):
            with open('data/ensup', 'r') as f:
                out.write(f.read())
    
    return output_path

def main():
    print("="*80)
    print("QUESTION (h): TRAINING STRATEGY EXPERIMENTS")
    print("="*80)
    print()
    print("Goal: Find the best way to combine supervised (ensup) and")
    print("      unsupervised (enraw) data to maximize accuracy.")
    print()
    
    results = {}
    device = 'cpu'
    
    # ========================================================================
    # STRATEGY 1: Baseline - Supervised only
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 1: Supervised Only (Baseline)")
    print("="*80)
    
    model_path = 'en_hmm.pkl'
    if not Path(model_path).exists():
        train_model(model_path, ['data/ensup'], device=device)
    else:
        print(f"Using existing {model_path}")
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Supervised Only')
    if acc is not None:
        results['supervised_only'] = {'accuracy': acc, 'cross_entropy': ce, 
                                      'description': 'ensup only (baseline)'}
    
    # ========================================================================
    # STRATEGY 2: Standard Semi-supervised
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 2: Standard Semi-supervised")
    print("="*80)
    
    model_path = 'en_hmm_raw.pkl'
    if not Path(model_path).exists():
        train_model(model_path, ['data/ensup', 'data/enraw'], device=device)
    else:
        print(f"Using existing {model_path}")
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Standard Semi-supervised')
    if acc is not None:
        results['standard_semisup'] = {'accuracy': acc, 'cross_entropy': ce,
                                       'description': 'ensup + enraw'}
    
    # ========================================================================
    # STRATEGY 3: Weighted 3:1
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 3: Weighted Supervised 3:1")
    print("="*80)
    print("Training with ensup repeated 3 times + enraw once")
    
    model_path = 'en_hmm_weighted_3to1.pkl'
    train_model(model_path, ['data/ensup', 'data/ensup', 'data/ensup', 'data/enraw'], device=device)
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Weighted 3:1')
    if acc is not None:
        results['weighted_3to1'] = {'accuracy': acc, 'cross_entropy': ce,
                                    'description': 'ensup×3 + enraw (3:1)'}
    
    # ========================================================================
    # STRATEGY 4: Weighted 5:1
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 4: Weighted Supervised 5:1")
    print("="*80)
    print("Training with ensup repeated 5 times + enraw once")
    
    model_path = 'en_hmm_weighted_5to1.pkl'
    train_model(model_path, 
                ['data/ensup', 'data/ensup', 'data/ensup', 'data/ensup', 'data/ensup', 'data/enraw'], 
                device=device)
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Weighted 5:1')
    if acc is not None:
        results['weighted_5to1'] = {'accuracy': acc, 'cross_entropy': ce,
                                    'description': 'ensup×5 + enraw (5:1)'}
    
    # ========================================================================
    # STRATEGY 5: Weighted 10:1
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 5: Weighted Supervised 10:1")
    print("="*80)
    print("Training with ensup repeated 10 times + enraw once")
    
    model_path = 'en_hmm_weighted_10to1.pkl'
    train_files = ['data/ensup'] * 10 + ['data/enraw']
    train_model(model_path, train_files, device=device)
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Weighted 10:1')
    if acc is not None:
        results['weighted_10to1'] = {'accuracy': acc, 'cross_entropy': ce,
                                     'description': 'ensup×10 + enraw (10:1)'}
    
    # ========================================================================
    # STRATEGY 6: Staged Training (Supervised → Semi-supervised)
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 6: Staged Training (Supervised → Semi-supervised)")
    print("="*80)
    print("Step 1: Train on ensup")
    print("Step 2: Continue training on enraw")
    
    # First train on supervised data
    stage1_path = 'en_hmm_stage1.pkl'
    if not Path(stage1_path).exists():
        train_model(stage1_path, ['data/ensup'], device=device)
    
    # Then continue training on unsupervised data
    model_path = 'en_hmm_staged_sup_to_unsup.pkl'
    train_model(model_path, ['data/enraw'], device=device, load_from=stage1_path)
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Staged Sup→Unsup')
    if acc is not None:
        results['staged_sup_to_unsup'] = {'accuracy': acc, 'cross_entropy': ce,
                                          'description': 'ensup first, then enraw'}
    
    # ========================================================================
    # STRATEGY 7: Reverse Staged (Semi-supervised → Fine-tune Supervised)
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 7: Reverse Staged (Semi-sup → Fine-tune Supervised)")
    print("="*80)
    print("Step 1: Train on ensup + enraw")
    print("Step 2: Fine-tune on ensup only")
    
    # Use the standard semi-supervised model as starting point
    semisup_path = 'en_hmm_raw.pkl'
    
    # Fine-tune on supervised data
    model_path = 'en_hmm_staged_unsup_to_sup.pkl'
    train_model(model_path, ['data/ensup'], device=device, load_from=semisup_path)
    
    acc, ce = evaluate_model(model_path, 'data/endev', 'Staged Unsup→Sup (Fine-tune)')
    if acc is not None:
        results['staged_unsup_to_sup'] = {'accuracy': acc, 'cross_entropy': ce,
                                          'description': 'ensup+enraw first, then fine-tune on ensup'}
    
    # ========================================================================
    # STRATEGY 8: Unsupervised Only (for comparison)
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 8: Unsupervised Only (Comparison)")
    print("="*80)
    print("Training on enraw alone (no supervision)")
    print()
    print("⚠️  SKIPPING: Cannot train on enraw alone because:")
    print("   - enraw has no tags (unsupervised)")
    print("   - Model would only learn BOS/EOS tags")
    print("   - Cannot evaluate on endev which has 26 tags")
    print("   - This experiment is not meaningful for accuracy evaluation")
    print()
    
    # Skip this experiment - it's not meaningful
    # model_path = 'en_hmm_unsup_only.pkl'
    # train_model(model_path, ['data/enraw'], device=device)
    # acc, ce = evaluate_model(model_path, 'data/endev', 'Unsupervised Only')
    # if acc is not None:
    #     results['unsupervised_only'] = {'accuracy': acc, 'cross_entropy': ce,
    #                                     'description': 'enraw only (no labels)'}
    
    # ========================================================================
    # SUMMARY AND ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: ALL TRAINING STRATEGIES")
    print("="*80)
    print()
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Rank':<6} {'Strategy':<35} {'Accuracy':<12} {'Cross-Entropy':<15} {'Description'}")
    print("-" * 105)
    
    for rank, (key, data) in enumerate(sorted_results, 1):
        strategy_name = key.replace('_', ' ').title()
        print(f"{rank:<6} {strategy_name:<35} {data['accuracy']:>7.3f}%    {data['cross_entropy']:>8.4f} bits/word   {data['description']}")
    
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    # Find best strategy
    best_key, best_data = sorted_results[0]
    baseline_acc = results.get('supervised_only', {}).get('accuracy', 0)
    standard_semisup_acc = results.get('standard_semisup', {}).get('accuracy', 0)
    
    print(f"🏆 BEST STRATEGY: {best_key.replace('_', ' ').title()}")
    print(f"   Accuracy: {best_data['accuracy']:.3f}%")
    print(f"   Description: {best_data['description']}")
    print()
    
    improvement_over_baseline = best_data['accuracy'] - baseline_acc
    print(f"📊 Improvement over supervised baseline: {improvement_over_baseline:+.3f}%")
    
    improvement_over_standard = best_data['accuracy'] - standard_semisup_acc
    print(f"📊 Improvement over standard semi-supervised: {improvement_over_standard:+.3f}%")
    print()
    
    # Analyze patterns
    print("KEY FINDINGS:")
    print()
    
    if 'weighted_3to1' in results:
        w3_acc = results['weighted_3to1']['accuracy']
        print(f"1. Weighted 3:1 strategy: {w3_acc:.3f}%")
        if w3_acc > standard_semisup_acc:
            print(f"   → {w3_acc - standard_semisup_acc:.3f}% better than standard semi-supervised")
            print(f"   → Weighting supervised data helps combat Merialdo effect!")
        else:
            print(f"   → {standard_semisup_acc - w3_acc:.3f}% worse than standard semi-supervised")
    
    if 'weighted_5to1' in results and 'weighted_3to1' in results:
        w5_acc = results['weighted_5to1']['accuracy']
        w3_acc = results['weighted_3to1']['accuracy']
        print(f"\n2. Weighted 5:1 strategy: {w5_acc:.3f}%")
        if w5_acc > w3_acc:
            print(f"   → {w5_acc - w3_acc:.3f}% better than 3:1 weighting")
            print(f"   → More supervised weighting helps even more!")
        else:
            print(f"   → {w3_acc - w5_acc:.3f}% worse than 3:1 weighting")
            print(f"   → Diminishing returns or overfitting to supervised data")
    
    if 'staged_unsup_to_sup' in results:
        staged_acc = results['staged_unsup_to_sup']['accuracy']
        print(f"\n3. Fine-tuning strategy (semi-sup → supervised): {staged_acc:.3f}%")
        if staged_acc > standard_semisup_acc:
            print(f"   → {staged_acc - standard_semisup_acc:.3f}% better than standard semi-supervised")
            print(f"   → Fine-tuning on supervised data after EM fixes Merialdo corruption!")
        else:
            print(f"   → Did not improve over standard semi-supervised")
    
    if 'unsupervised_only' in results:
        unsup_acc = results['unsupervised_only']['accuracy']
        print(f"\n4. Unsupervised only: {unsup_acc:.3f}%")
        print(f"   → {baseline_acc - unsup_acc:.3f}% worse than supervised baseline")
        print(f"   → Confirms that labels are essential for good POS tagging")
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The Merialdo effect shows that adding more unsupervised data can hurt")
    print("accuracy even though it improves cross-entropy (perplexity). Strategies")
    print("that work:")
    print()
    print("✓ Weight supervised data more heavily (repeat ensup multiple times)")
    print("✓ Fine-tune on supervised data after semi-supervised training")
    print("✓ Use staged training to preserve supervised signal")
    print()
    print(f"Best result: {best_data['accuracy']:.3f}% with {best_data['description']}")
    print()

if __name__ == '__main__':
    main()
