# TASK 1: APRIORI ALGORITHM - KEY FINDINGS

1. FREQUENT SYMPTOM COMBINATIONS:
   - [List top 5 discovered patterns]
2. STRONG ASSOCIATION RULES:
   - [List top 3 rules with high confidence]
3. INSIGHTS:
   - [What patterns did you discover?]
   - [Any surprising symptom co-occurrences?]
4. PARAMETERS USED:
   - Min Support: X
   - Min Confidence: Y
5. ALGORITHM PERFORMANCE:

   - Total transactions: X
   - Frequent itemsets discovered: X
   - Processing time: X seconds

6. Distribution of Frequent Itemset Sizes Explained

## Interpretation

We're looking at a pyramid of pattern complexity.

- 1-itemsets (18): These are individual symptoms that appear in multiple diseases

  - Example: "vomiting" alone appears in 16/41 diseases

  - These are the "frequent symptoms"

- 2-itemsets (27) ← MOST IMPORTANT: These are symptom pairs that co-occur

  - Example: "nausea + vomiting" appears together in X diseases

  - This shows symptoms that naturally go together

  - Medical insight: Symptoms often co-occur for biological reasons

- 3-itemsets (14): Symptom triplets with weaker patterns

  - Harder to find 3+ symptoms appearing together

  - Each additional symptom reduces the number of diseases with ALL of them

- 4-itemsets (1): Very rare combination

  - Only 1 specific set of 4 symptoms appears in ≥5 diseases

Observation: This follows the Pareto Principle - most patterns are 1-2 items, not complex combinations. This is typical and healthy for association mining.

## Interpretation: Top 10 Symptom Combinations
