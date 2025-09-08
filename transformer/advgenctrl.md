# Advanced Generation Control

## Token Selection and Sampling

- The selection of next token at each step can be controlled through various parameters:

1. Raw Logits: the initial output probabilities for each token.

2. Temperature: controls randomness in selection

3. Top-p (Nucleus sampling): Filters to top tokens making up X% of probability mass

4. Top-k filtering: limits selection to k most likely tokens.

## Controlling repetition

- penalize frequent tokens
- penalise tokens already present

## Length control and stop sequences

- we can control generation length and specify when to stop.
