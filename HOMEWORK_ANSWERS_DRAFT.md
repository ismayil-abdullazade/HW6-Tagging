# Homework 6 Answers (Draft)

## Question 2(a): Why does Algorithm 1 initialize α_BOS(0) to 1, and Algorithm 4 initialize β_EOS(n+1) to 1?

**Answer**: 
- **α_BOS(0) = 1**: The alpha value represents the total probability of all ways to generate the prefix up to position j ending in a particular tag. At position 0, we know with certainty that we're at BOS (the beginning of sentence marker), so there's only one "path" with probability 1. All other tags at position 0 have α = 0 since we can't be in any other state at the start.

- **β_EOS(n+1) = 1**: The beta value represents the probability of generating the suffix from position j onward, given that we're in a particular state at position j. At the final position (n+1), we must be in the EOS (end of sentence) state, and there's nothing left to generate, so the probability is 1. This is the base case for the backward recursion.

These initializations are boundary conditions that make the dynamic programming recurrences work correctly.

## Question 2(b): If you train on a sup file and then evaluate on a held-out raw file, you'll get lower perplexity than if you evaluate on a held-out dev file. Why is that? Which perplexity do you think is more important and why?

**Answer**:
The **raw** file contains untagged sentences, so when computing perplexity we only evaluate p(words), summing over all possible tag sequences. The **dev** file contains tagged sentences, so we evaluate p(tags, words) for the specific observed tags.

Since p(words) = Σ_tags p(tags, words), and we're summing over many possible tag sequences (most of which are less probable than the gold tags), the joint probability p(tags, words) for the specific gold tagging will be lower than p(words). Lower probability means higher perplexity for the dev file.

**The dev file perplexity is more important** because:
1. It measures how well the model predicts the correct linguistic structure (tags + words)
2. It better reflects the true evaluation task (tagging accuracy)
3. The raw file perplexity doesn't penalize the model for getting tags wrong

## Question 2(c): V includes the word types from sup and raw (plus oov). Why not from dev as well?

**Answer**:
The vocabulary should only include words from the **training data** (sup and raw), not from the development or test data. This is because:

1. **Realistic evaluation**: In a real-world scenario, we won't see the test data beforehand. Including dev vocabulary would give us unfair advantage.
2. **Test OOV handling**: We need to test how well the model handles out-of-vocabulary (OOV) words. If we include dev words in the vocabulary, we can't properly evaluate the model's ability to handle unseen words.
3. **No peeking**: Including dev vocabulary would be a form of "peeking" at the test set, which violates standard machine learning evaluation practices.

Words from the dev set that don't appear in training will be mapped to the OOV token, allowing us to measure performance on novel words.

## Question 2(d): Did the iterations of semi-supervised training help or hurt overall tagging accuracy? How about tagging accuracy on known, seen, and novel words (respectively)?

**Answer**: [TO BE FILLED AFTER RUNNING EXPERIMENTS]
- Overall accuracy: 
- Known words (from supervised training): 
- Seen words (from all training): 
- Novel words (OOV): 

## Question 2(e): Explain in a few clear sentences why the semi-supervised approach might sometimes help. How does it get additional value out of the enraw file?

**Answer**:
The semi-supervised approach can help because:

1. **Better transition estimates**: The enraw file provides many examples of natural tag sequences, helping the model learn better transition probabilities p(tag | prev_tag). Even without knowing the actual tags, the EM algorithm can infer likely tag sequences based on the emission patterns.

2. **Better emission estimates**: Frequent words that appear in both supervised and unsupervised data get more training examples, leading to more robust emission probability estimates p(word | tag).

3. **Smoothing effect**: The additional unsupervised data can help smooth the probability distributions, potentially reducing overfitting to the smaller supervised dataset.

4. **Context disambiguation**: For words that appear in limited contexts in the supervised data, the unsupervised data provides additional contexts that help the model learn when each tag is appropriate.

The EM algorithm extracts value from enraw by using the current model to compute expected tag counts (E-step), then updating parameters based on these expectations (M-step). Over iterations, the model's "soft" labeling of the unsupervised data improves.

## Question 2(f): Suggest at least two reasons to explain why the semi-supervised approach didn't always help.

**Answer**:

1. **Error reinforcement**: If the supervised model makes systematic errors when tagging the unsupervised data, the EM algorithm may reinforce these errors rather than correct them. The "rich get richer" effect means that incorrect patterns in the initial supervised model can be amplified during semi-supervised training.

2. **Domain mismatch**: If the unsupervised data (enraw) comes from a different domain or has different characteristics than the supervised data (ensup), the model may learn patterns that don't generalize well to the development set. This could hurt performance if the dev set is more similar to the supervised training data.

3. **Local optima**: EM is not guaranteed to find the global optimum. Semi-supervised training might converge to a worse local optimum than supervised training alone, especially if the initialization from supervised training isn't ideal.

4. **Overfitting to noise**: The unsupervised data might contain noise, unusual constructions, or rare patterns that the model overfits to, reducing its performance on more standard text in the development set.

## Question 2(g): How does your bigram HMM tagger compare to a baseline unigram HMM tagger? Consider both accuracy and cross-entropy. Does it matter whether you use enraw?

**Answer**: [TO BE FILLED AFTER RUNNING EXPERIMENTS]

Expected observations:
- Bigram should have **better accuracy** because it uses context (previous tag)
- Bigram should have **lower cross-entropy** (better perplexity) because it's a more powerful model
- The difference should be **more pronounced with enraw** because the bigram model can learn better transition patterns from the unsupervised data

Results:
- Unigram supervised: 
- Bigram supervised: 
- Unigram semi-supervised: 
- Bigram semi-supervised: 

## Question 4(a): Compare the CRF to the HMM, training on ensup and evaluating on endev.

**Answer**: [TO BE FILLED AFTER RUNNING EXPERIMENTS]

Expected differences:
- **Training objective**: HMM maximizes joint probability p(tags, words), CRF maximizes conditional probability p(tags | words)
- **Cross-entropy**: CRF reports conditional cross-entropy (lower numbers), HMM reports joint cross-entropy (higher numbers) - not directly comparable
- **Accuracy**: Should be similar with only transition/emission features, possibly slight CRF advantage due to discriminative training
- **Speed**: HMM training (EM) might be faster than CRF training (SGD)

Results:
- HMM accuracy: 
- CRF accuracy: 
- HMM cross-entropy (joint): 
- CRF cross-entropy (conditional): 

## Question 4(b): What happens when you include enraw in the training data for your CRF? Explain.

**Answer**:
CRF training should **not benefit** from untagged data (enraw) because:

1. **Discriminative model**: The CRF is a discriminative model that learns p(tags | words). It requires labeled data to compute the gradient, which is the difference between observed tag counts and expected tag counts under the model.

2. **No gradient from unlabeled data**: For completely untagged sentences, the conditional probability p(tags | words) = 1 (we can pick any tagging), so there's no information to learn from. The gradient would be zero.

3. **Supervised training only**: Unlike the HMM's generative EM algorithm which can use unlabeled data, the CRF's discriminative SGD algorithm needs tag labels to compute meaningful gradients.

[Results after experiment will confirm this expectation]

---

## Implementation Notes

### Numerical Stability
All algorithms implemented in log-space to avoid underflow:
- Forward algorithm: computes log α values, uses `torch.logsumexp()`
- Backward algorithm: computes log β values, uses `torch.logsumexp()`
- Count accumulation: computes log posteriors, only adds if > -100 to avoid underflow

### Testing
- ✅ Ice cream dataset: Matches spreadsheet values
- ✅ Supervised training: Converges properly
- ✅ CRF training: Works correctly with gradient ascent
