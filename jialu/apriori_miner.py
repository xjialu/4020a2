"""
apriori_miner.py

This module implements the Apriori algorithm for frequent itemset mining.

Algorithm Reference:
- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
- Concept: Generate candidate itemsets and prune based on minimum support threshold
"""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


class AprioriMiner:
    """
    Apriori algorithm implementation for frequent itemset mining.
    """

    def __init__(self, min_support=0.1):
        """
        Initialize the Apriori miner.

        Args:
            min_support (float): Minimum support threshold (0-1)
                               Support = number of transactions containing itemset / total transactions
        """
        self.min_support = min_support
        self.transactions = []
        self.frequent_itemsets = {}  # {k: [(itemset, support), ...]}
        self.association_rules = []
        self.all_frequent_itemsets = []

    def fit(self, transactions):
        """
        Fit the Apriori algorithm on transaction data.

        Args:
            transactions (list): List of transactions, where each transaction is a frozenset of items
        """
        self.transactions = transactions
        self.frequent_itemsets = {}
        self.association_rules = []
        self.all_frequent_itemsets = []

        # Calculate minimum support count (absolute number)
        min_support_count = int(np.ceil(self.min_support * len(transactions)))

        print(
            f"Fitting Apriori with min_support={self.min_support} (count={min_support_count})"
        )
        print(f"Total transactions: {len(transactions)}")

        # Step 1: Generate 1-itemsets
        print("Generating 1-itemsets...")
        one_itemsets = self._generate_one_itemsets(min_support_count)

        if not one_itemsets:
            print("No frequent 1-itemsets found. Try lowering min_support.")
            return

        self.frequent_itemsets[1] = one_itemsets
        current_itemsets = one_itemsets
        k = 1

        # Step 2: Iteratively generate k-itemsets
        while len(current_itemsets) > 1:
            k += 1
            print(f"Generating {k}-itemsets...")

            # Generate candidates
            candidates = self._generate_candidates(current_itemsets, k)

            if not candidates:
                break

            # Prune candidates
            frequent_k_itemsets = self._prune_candidates(candidates, min_support_count)

            if not frequent_k_itemsets:
                print(f"  Found 0 frequent {k}-itemsets")
                break

            self.frequent_itemsets[k] = frequent_k_itemsets
            current_itemsets = frequent_k_itemsets

            print(f"  Found {len(frequent_k_itemsets)} frequent {k}-itemsets")

        # Collect all frequent itemsets
        for k in sorted(self.frequent_itemsets.keys()):
            self.all_frequent_itemsets.extend(self.frequent_itemsets[k])

        print(f"Total frequent itemsets: {len(self.all_frequent_itemsets)}")

    def _generate_one_itemsets(self, min_support_count):
        """
        Generate frequent 1-itemsets.

        Args:
            min_support_count (int): Minimum support count threshold

        Returns:
            list: List of (frozenset, support) tuples for frequent 1-itemsets
        """
        # Count occurrences of each item
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filter by minimum support
        frequent_1_itemsets = [
            (frozenset([item]), count)
            for item, count in item_counts.items()
            if count >= min_support_count
        ]

        print(f"  Found {len(frequent_1_itemsets)} frequent 1-itemsets")

        return sorted(frequent_1_itemsets, key=lambda x: x[1], reverse=True)

    def _generate_candidates(self, current_itemsets, k):
        """
        Generate candidate k-itemsets from (k-1)-itemsets using join step.

        Reference: Agrawal & Srikant (1994) - F(k-1) join F(k-1)

        Args:
            current_itemsets (list): Current frequent itemsets
            k (int): Size of itemsets to generate

        Returns:
            list: List of candidate k-itemsets as frozensets
        """
        # Extract just the itemsets (without support counts)
        itemsets_only = [itemset for itemset, _ in current_itemsets]
        itemsets_set = set(itemsets_only)  # For O(1) lookup

        # Generate candidates by combining (k-1)-itemsets
        candidates = set()
        for i in range(len(itemsets_only)):
            for j in range(i + 1, len(itemsets_only)):
                union = itemsets_only[i] | itemsets_only[j]

                # Only keep if union has exactly k items
                # if len(union) == k:
                #     candidates.add(union)
                    
                if len(union) == k:
                    # All (k-1)-subsets must be frequent
                    valid = True
                    for subset in combinations(union, k - 1):
                        if frozenset(subset) not in itemsets_set:
                            valid = False
                            break

                    if valid:
                        candidates.add(union)

        return list(candidates)

    def _prune_candidates(self, candidates, min_support_count):
        """
        Prune candidates by support (Apriori pruning step).

        Args:
            candidates (list): List of candidate itemsets
            min_support_count (int): Minimum support count threshold

        Returns:
            list: List of (frozenset, support) tuples for frequent itemsets
        """
        # Count support for each candidate
        candidate_counts = defaultdict(int)
        for transaction in self.transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        # Filter by minimum support
        frequent_itemsets = [
            (itemset, count)
            for itemset, count in candidate_counts.items()
            if count >= min_support_count
        ]

        return sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)

    def generate_rules(self, min_confidence=0.5):
        """
        Generate association rules from frequent itemsets.

        For each frequent itemset X with |X| > 1:
            For each A c X:
                If confidence(A -> X-A) >= min_confidence:
                    Generate rule A -> X-A

        Args:
            min_confidence (float): Minimum confidence threshold (0-1)
                                   Confidence = support(A u B) / support(A)

        Returns:
            list: List of association rules
        """
        self.association_rules = []

        # Create support dictionary for faster lookup
        support_dict = {
            itemset: support for itemset, support in self.all_frequent_itemsets
        }

        # Generate rules from itemsets with 2 or more items
        for itemset, support_AB in self.all_frequent_itemsets:
            if len(itemset) < 2:
                continue

            # Generate all possible antecedents
            for antecedent_size in range(1, len(itemset)):
                for antecedent in combinations(sorted(itemset), antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent

                    # Look up support values
                    if antecedent not in support_dict:
                        continue

                    support_A = support_dict[antecedent]

                    # Calculate confidence and lift
                    confidence = support_AB / support_A

                    if confidence >= min_confidence:
                        # Calculate lift
                        support_B = support_dict.get(consequent, 0)
                        if support_B > 0:
                            lift = (support_AB * len(self.transactions)) / (
                                support_A * support_B
                            )
                        else:
                            lift = 0

                        rule = {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": support_AB / len(self.transactions),
                            "confidence": confidence,
                            "lift": lift,
                        }
                        self.association_rules.append(rule)

        # Sort by confidence
        self.association_rules.sort(key=lambda x: x["confidence"], reverse=True)

        return self.association_rules

    def get_frequent_itemsets(self, k=None):
        """
        Get frequent itemsets.

        Args:
            k (int, optional): Get itemsets of size k. If None, get all.

        Returns:
            list: List of (itemset, support_count) tuples
        """
        if k is None:
            return self.all_frequent_itemsets
        return self.frequent_itemsets.get(k, [])

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("APRIORI ALGORITHM - RESULTS SUMMARY")
        print("=" * 70 + "\n")

        print(f"Min Support: {self.min_support}")
        print(f"Total Transactions: {len(self.transactions)}")
        print(f"Total Frequent Itemsets: {len(self.all_frequent_itemsets)}\n")

        print("Itemsets by Size:")
        for k in sorted(self.frequent_itemsets.keys()):
            count = len(self.frequent_itemsets[k])
            print(f"  {k}-itemsets: {count}")

        print(f"\nAssociation Rules: {len(self.association_rules)}\n")

        if self.association_rules:
            print("Top 10 Rules by Confidence:")
            for i, rule in enumerate(self.association_rules[:10], 1):
                print(f"\n  {i}. {self._format_rule(rule)}")
                print(f"     Confidence: {rule['confidence']:.3f}")
                print(f"     Support: {rule['support']:.3f}")
                print(f"     Lift: {rule['lift']:.3f}")

    def _format_rule(self, rule):
        """Format a rule as a readable string."""
        antecedent = ", ".join(sorted(rule["antecedent"]))
        consequent = ", ".join(sorted(rule["consequent"]))
        return f"{{{antecedent}}} -> {{{consequent}}}"
