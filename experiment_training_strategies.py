#!/usr/bin/env python3
"""
Experiment with different training strategies for semi-supervised learning.
This script tests various ways of incorporating enraw data into HMM training.
"""

import subprocess
import json
import re
from datetime import datetime

print("="*80)
print("TRAINING STRATEGY EXPERIMENTS")
print("Testing different ways to incorporate enraw data into HMM training")
print("="*80)

# Store all results
results = {}

def run_training(name, command, description):
    """Run a training command and capture the results."""
    print(f"\n{'='*80}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
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
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/enraw --model en_hmm_unsup.pkl --max_steps 50 --device cpu",
    "Fully unsupervised: Train on enraw alone (no supervised data)"
)

# Strategy 2: Combined corpus (ensup + enraw) - this is what we already have
print("\n" + "="*80)
print("Strategy 2: Combined corpus (ensup + enraw)")
print("This is en_hmm_raw.pkl which we already trained: 88.283% accuracy")
print("="*80)

# Strategy 3: Weighted supervised 3:1 (ensup + ensup + ensup + enraw)
# We need to create a temporary file with ensup repeated 3 times
print("\n" + "="*80)
print("Strategy 3: Preparing weighted corpus (3x supervised weight)")
print("="*80)
subprocess.run(
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && cat data/ensup data/ensup data/ensup > data/ensup3x",
    shell=True
)
print("✓ Created data/ensup3x (ensup repeated 3 times)")

run_training(
    "strategy_3_weighted_3to1",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/ensup3x data/enraw --model en_hmm_weighted3.pkl --max_steps 50 --device cpu",
    "Weighted 3:1: Train on (ensup×3) + enraw to weight supervised data more heavily"
)

# Strategy 4: Weighted supervised 2:1
subprocess.run(
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && cat data/ensup data/ensup > data/ensup2x",
    shell=True
)
print("✓ Created data/ensup2x (ensup repeated 2 times)")

run_training(
    "strategy_4_weighted_2to1",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/ensup2x data/enraw --model en_hmm_weighted2.pkl --max_steps 50 --device cpu",
    "Weighted 2:1: Train on (ensup×2) + enraw for moderate supervised weighting"
)

# Strategy 5: Staged training - unsupervised first, then supervised fine-tuning
print("\n" + "="*80)
print("Strategy 5: Staged training (unsupervised first, then supervised)")
print("="*80)
# First train on enraw alone
run_training(
    "strategy_5_stage1",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/enraw --model en_hmm_stage5_1.pkl --max_steps 30 --device cpu",
    "Stage 1: Pre-train on enraw (unsupervised)"
)
# Then continue training on ensup
run_training(
    "strategy_5_stage2",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/ensup --model en_hmm_stage5_2.pkl --checkpoint en_hmm_stage5_1.pkl --max_steps 30 --device cpu",
    "Stage 2: Fine-tune on ensup (supervised) starting from stage 1"
)

# Strategy 6: Staged training - supervised first, then unsupervised
print("\n" + "="*80)
print("Strategy 6: Staged training (supervised first, then unsupervised)")
print("="*80)
# First train on ensup (use existing en_hmm.pkl)
# Then continue training on enraw
run_training(
    "strategy_6_continue",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/enraw --model en_hmm_stage6.pkl --checkpoint en_hmm.pkl --max_steps 30 --device cpu",
    "Continue training on enraw starting from supervised model (en_hmm.pkl)"
)

# Strategy 7: Staged training - supervised, then combined
print("\n" + "="*80)
print("Strategy 7: Staged training (supervised first, then combined)")
print("="*80)
run_training(
    "strategy_7_combined",
    "cd /mnt/c/Users/Asus1/Downloads/HW-TAG && python code/tag.py data/endev --train data/ensup data/enraw --model en_hmm_stage7.pkl --checkpoint en_hmm.pkl --max_steps 30 --device cpu",
    "Continue training on ensup+enraw starting from supervised model"
)

# Print summary
print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

print("\n{:<35} {:>10} {:>10} {:>10} {:>12}".format(
    "Strategy", "Accuracy", "Known", "Novel", "Cross-ent"
))
print("-" * 80)

# Add baseline results
print("{:<35} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
    "Baseline: Supervised only",
    90.455, 96.786, 24.858, 10.8059
))
print("{:<35} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
    "Baseline: Semi-supervised",
    88.283, 92.440, 26.684, 9.6054
))

for name, metrics in sorted(results.items()):
    if 'error' not in metrics and metrics['accuracy'] is not None:
        print("{:<35} {:>9.3f}% {:>9.3f}% {:>9.3f}% {:>12.4f}".format(
            name.replace('_', ' ').title(),
            metrics['accuracy'],
            metrics['known_accuracy'],
            metrics['novel_accuracy'],
            metrics['cross_entropy']
        ))
    elif 'error' in metrics:
        print("{:<35} {:>10}".format(
            name.replace('_', ' ').title(),
            f"ERROR: {metrics['error']}"
        ))

# Save detailed results to JSON
with open('training_strategies_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Detailed results saved to training_strategies_results.json")

# Analysis and recommendations
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
Expected patterns to observe:

1. **Unsupervised only (Strategy 1)**: Should have VERY poor accuracy (~30-40%)
   - Tags get repurposed arbitrarily without supervised signal
   - Demonstrates why we need at least some labeled data

2. **Weighted supervised (Strategies 3-4)**: Should improve over baseline semi-supervised
   - More weight on supervised data reduces Merialdo effect
   - 3:1 weighting should be better than 2:1
   - Should recover some of the accuracy lost in semi-supervised training

3. **Staged: Unsup → Sup (Strategy 5)**: Likely poor performance
   - Starting from random/unsupervised tags makes supervised fine-tuning harder
   - The tag meanings learned in stage 1 don't match supervised tag meanings

4. **Staged: Sup → Unsup (Strategy 6)**: May cause degradation
   - Starting from good supervised model, unsupervised data may cause drift
   - Similar to baseline semi-supervised but possibly worse

5. **Staged: Sup → Combined (Strategy 7)**: Should help
   - Warm start from good supervised model
   - Combined training may find better local optimum than starting from scratch

Best strategy is likely: Weighted 3:1 or staged Sup → Combined
""")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE")
print("="*80)
