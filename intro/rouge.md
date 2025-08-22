# ROUGE

- Rouge or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in NLP. 

- It compares an automatically produced summary or translation against a reference or a set of references(human-produced) summary or translation.

- **ROUGE-N**(N-gram) scoring
- **ROUGE-L**(Longest Common Subsequence) scoring
- Text Normalization
- Bootstrap resampling for confidence interval calculation
- Optional Porter stemming

## 2 flavours of ROUGE-L

1. Sentence Level: compute longest common subsequence between two pieces of text.

2. summary-level: Newlines in the text are interpreted as sentence boundaries and the LCS is computed between each pair of reference and candidate sentences
