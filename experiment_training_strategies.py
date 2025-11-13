#!/usr/bin/env python3
"""
Experiment with different training strategies for semi-supervised learning.
This script tests various ways of incorporating enraw data into HMM training.
All paths are relative to the HW-TAG directory.
"""

import subprocess
import json
import re
from datetime import datetime
import os

print("="*80)
print("TRAINING STRATEGY EXPERIMENTS - Question 2(h) Extra Credit")
print("Testing different ways to incorporate enraw data into HMM training")
print("="*80)

# Store all results
results = {}

def run_training(name, train_args, model_name, description, checkpoint=None):
    """Run a training command and capture the results."""
    print(f"\n{'='*80}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
    
    # Build command with relative paths
    cmd_parts = ["python", "code/tag.py", "data/endev"]
    
    # Add training data
    if train_args:
        cmd_parts.append("--train")
        cmd_parts.extend(train_args)
    
    # Add model
    cmd_parts.extend(["--model", model_name])
    
    # Add checkpoint if continuing training
    if checkpoint:
        cmd_parts.extend(["--checkpoint", checkpoint])
    
    # Add other options
    cmd_parts.extend(["--max_steps", "50", "--device", "cpu"])
    
    command = " ".join(cmd_parts)
    print(f"Command: {command}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        output = result.stdout + result.stderr
        print(output)
        
        # Parse accuracy and metrics from output
        accuracy_match = re.search(r'Overall accuracy:\s+([\d.]+)%', output)
        known_match = re.search(r'Known-word accuracy:\s+([\d.]+)%', output)
        novel_match = re.search(r'Novel-word accuracy:\s+([\d.]+)%', output)
        ce_match = re.search(r'Cross-entropy:\s+([\d.]+)', output)
        perp_match = re.search(r'Perplexity:\s+([\d.]+)', output)
        
        metrics = {
            'accuracy': float(accuracy_match.group(1)) if accuracy_match else None,
            'known_accuracy': float(known_match.group(1)) if known_match else None,
            'novel_accuracy': float(novel_match.group(1)) if novel_match else None,
            'cross_entropy': float(ce_match.group(1)) if ce_match else None,
            'perplexity': float(perp_match.group(1)) if perp_match else None,
            'description': description,
            'command': command,
            'output': output
        }
        
        results[name] = metrics
        print(f"\n✓ {name} completed:")
        if metrics['accuracy']:
            print(f"  Accuracy: {metrics['accuracy']}%")
            print(f"  Known: {metrics['known_accuracy']}%, Novel: {metrics['novel_accuracy']}%")
            print(f"  Cross-entropy: {metrics['cross_entropy']}, Perplexity: {metrics['perplexity']}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"✗ {name} timed out after 10 minutes")
        results[name] = {'error': 'timeout', 'description': description}
        return None
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        results[name] = {'error': str(e), 'description': description}
        return None

# Baseline: Already trained
print("\n" + "="*80)
print("BASELINE MODELS (already trained)")
print("="*80)
print("1. Supervised only (en_hmm.pkl): 90.455% accuracy")
print("2. Semi-supervised (en_hmm_raw.pkl): 88.283% accuracy")

# Strategy 1: Train on enraw alone (fully unsupervised)
run_training(
    "strategy_1_unsupervised",
    ["data/enraw"],
    "en_hmm_unsup.pkl",
    "Fully unsupervised: Train on enraw alone (no supervised data)"
)

# Strategy 2: Combined corpus (ensup + enraw) - this is what we already have
print("\n" + "="*80)
print("Strategy 2: Combined corpus (ensup + enraw)")
print("This is en_hmm_raw.pkl which we already trained: 88.283% accuracy")
print("="*80)

# Strategy 3: Weighted supervised 3:1 (ensup + ensup + ensup + enraw)
print("\n" + "="*80)
print("Strategy 3: Preparing weighted corpus (3x supervised weight)")
print("="*80)
if not os.path.exists("data/ensup3x"):
    subprocess.run(
        "cat data/ensup data/ensup data/ensup > data/ensup3x",
        shell=True
    )
    print("✓ Created data/ensup3x (ensup repeated 3 times)")
else:
    print("✓ data/ensup3x already exists")

run_training(
    "strategy_3_weighted_3to1",
    ["data/ensup3x", "data/enraw"],
    "en_hmm_weighted3.pkl",
    "Weighted 3:1: Train on (ensup×3) + enraw to weight supervised data more heavily"
)

# Strategy 4: Weighted supervised 2:1
if not os.path.exists("data/ensup2x"):
    subprocess.run(
        "cat data/ensup data/ensup > data/ensup2x",
        shell=True
    )
    print("✓ Created data/ensup2x (ensup repeated 2 times)")
else:
    print("✓ data/ensup2x already exists")

run_training(
    "strategy_4_weighted_2to1",
    ["data/ensup2x", "data/enraw"],
    "en_hmm_weighted2.pkl",
    "Weighted 2:1: Train on (ensup×2) + enraw for moderate supervised weighting"
)

# Strategy 5: Weighted supervised 5:1
if not os.path.exists("data/ensup5x"):
    subprocess.run(
        "cat data/ensup data/ensup data/ensup data/ensup data/ensup > data/ensup5x",
        shell=True
    )
    print("✓ Created data/ensup5x (ensup repeated 5 times)")
else:
    print("✓ data/ensup5x already exists")

run_training(
    "strategy_5_weighted_5to1",
    ["data/ensup5x", "data/enraw"],
    "en_hmm_weighted5.pkl",
    "Weighted 5:1: Train on (ensup×5) + enraw for strong supervised weighting"
)

# Strategy 6: Staged training - unsupervised first, then supervised fine-tuning
print("\n" + "="*80)
print("Strategy 6: Staged training (unsupervised first, then supervised)")
print("="*80)
# First train on enraw alone
run_training(
    "strategy_6_stage1",
    ["data/enraw"],
    "en_hmm_stage6_1.pkl",
    "Stage 1: Pre-train on enraw (unsupervised)",
)
# Then continue training on ensup
run_training(
    "strategy_6_stage2",
    ["data/ensup"],
    "en_hmm_stage6_2.pkl",
    "Stage 2: Fine-tune on ensup (supervised) starting from stage 1",
    checkpoint="en_hmm_stage6_1.pkl"
)

# Strategy 7: Staged training - supervised first, then unsupervised
print("\n" + "="*80)
print("Strategy 7: Staged training (supervised first, then unsupervised)")
print("="*80)
# Use existing en_hmm.pkl and continue on enraw
run_training(
    "strategy_7_continue",
    ["data/enraw"],
    "en_hmm_stage7.pkl",
    "Continue training on enraw starting from supervised model (en_hmm.pkl)",
    checkpoint="en_hmm.pkl"
)

# Strategy 8: Staged training - supervised, then combined
print("\n" + "="*80)
print("Strategy 8: Staged training (supervised first, then combined)")
print("="*80)
run_training(
    "strategy_8_combined",
    ["data/ensup", "data/enraw"],
    "en_hmm_stage8.pkl",
    "Continue training on ensup+enraw starting from supervised model",
    checkpoint="en_hmm.pkl"
)

# Print summary
print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

print("\n{:<40} {:>10} {:>10} {:>10} {:>12}".format(
    "Strategy", "Accuracy", "Known", "Novel", "Cross-ent"
))
print("-" * 85)

# Add baseline results
print("{:<40} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
    "Baseline: Supervised only",
    90.455, 96.786, 24.858, 10.8059
))
print("{:<40} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
    "Baseline: Semi-supervised (ensup+enraw)",
    88.283, 92.440, 26.684, 9.6054
))
print("-" * 85)

# Print experiment results
for name, metrics in results.items():
    if 'error' in metrics:
        print("{:<40} {:>10}".format(name, f"ERROR: {metrics['error']}"))
    elif metrics.get('accuracy'):
        print("{:<40} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
            name,
            metrics['accuracy'],
            metrics['known_accuracy'],
            metrics['novel_accuracy'],
            metrics['cross_entropy']
        ))

# Save detailed results to JSON
output_file = f"training_strategies_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Detailed results saved to: {output_file}")

# Analysis and recommendations
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

if results:
    # Find best strategy by accuracy
    valid_results = {k: v for k, v in results.items() if v.get('accuracy')}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best performing strategy: {best[0]}")
        print(f"   Accuracy: {best[1]['accuracy']}%")
        print(f"   Description: {best[1]['description']}")
        
        # Compare to baselines
        if best[1]['accuracy'] > 90.455:
            print(f"\n✨ This beats supervised-only by {best[1]['accuracy'] - 90.455:.3f}%!")
        elif best[1]['accuracy'] > 88.283:
            print(f"\n✓ This beats standard semi-supervised by {best[1]['accuracy'] - 88.283:.3f}%")
        else:
            print(f"\n⚠ This underperforms standard semi-supervised by {88.283 - best[1]['accuracy']:.3f}%")

print("\n" + "="*80)
print("Experiment complete! Run times will be logged.")
print("="*80)
