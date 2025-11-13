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
import pandas as pd


class AprioriMiner:
    """
    Apriori algorithm implementation for frequent itemset mining.
    """

    def __init__(self, min_support=0.1):
        """
        Initialize the Apriori miner.

        Args:
            min_support (float): Minimum support threshold (0-1)
        """
        self.min_support = min_support
        self.transactions = []
        self.frequent_itemsets = {}  # {k: [(itemset, count), ...]}
        self.association_rules = []
        self.all_frequent_itemsets = []
        self.n_transactions = 0

    def fit(self, transactions):
        """
        Fit the Apriori algorithm on transaction data.

        Args:
            transactions (list): List of frozensets (transactions)
        """
        self.transactions = transactions
        self.n_transactions = len(transactions)
        self.frequent_itemsets = {}
        self.association_rules = []
        self.all_frequent_itemsets = []

        # Calculate minimum support count (absolute number)
        min_support_count = int(np.ceil(self.min_support * self.n_transactions))

        print(
            f"Fitting Apriori with min_support={self.min_support} "
            f"(count={min_support_count})"
        )
        print(f"Total transactions: {self.n_transactions}")

        # Step 1: Generate 1-itemsets
        print("Generating 1-itemsets...")
        one_itemsets = self._generate_one_itemsets(min_support_count)

        if not one_itemsets:
            print("No frequent 1-itemsets found. Try lowering min_support.")
            return self.all_frequent_itemsets

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
            frequent_k_itemsets = self._prune_candidates(
                candidates, min_support_count
            )

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
        return self.all_frequent_itemsets

    def _generate_one_itemsets(self, min_support_count):
        """Generate frequent 1-itemsets."""
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
        Generate candidate k-itemsets using join and prune steps.
        """
        # Extract itemsets and create lookup set
        itemsets_only = [itemset for itemset, _ in current_itemsets]
        itemsets_set = set(itemsets_only)

        candidates = set()

        # Join step: combine (k-1)-itemsets
        for i in range(len(itemsets_only)):
            for j in range(i + 1, len(itemsets_only)):
                union = itemsets_only[i] | itemsets_only[j]

                # Only keep if union has exactly k items
                if len(union) == k:
                    # Prune step: all (k-1)-subsets must be frequent
                    valid = True
                    for subset in combinations(union, k - 1):
                        if frozenset(subset) not in itemsets_set:
                            valid = False
                            break

                    if valid:
                        candidates.add(union)

        return list(candidates)

    def _prune_candidates(self, candidates, min_support_count):
        """Prune candidates by counting support."""
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

    def generate_rules(self, min_confidence=0.5, min_lift=1.0):
        """
        Generate association rules from frequent itemsets.

        Args:
            min_confidence (float): Minimum confidence threshold
            min_lift (float): Minimum lift threshold

        Returns:
            list: Association rules
        """
        self.association_rules = []

        # Create support dictionaries (both count and ratio)
        support_count_dict = {
            itemset: count for itemset, count in self.all_frequent_itemsets
        }
        support_ratio_dict = {
            itemset: count / self.n_transactions
            for itemset, count in self.all_frequent_itemsets
        }

        # Generate rules from itemsets with 2+ items
        for itemset, count_AB in self.all_frequent_itemsets:
            if len(itemset) < 2:
                continue

            # Try all possible antecedent sizes
            for antecedent_size in range(1, len(itemset)):
                for antecedent in combinations(sorted(itemset), antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent

                    # Get support counts
                    if antecedent not in support_count_dict:
                        continue

                    count_A = support_count_dict[antecedent]
                    count_B = support_count_dict.get(consequent, 0)

                    if count_B == 0:
                        continue

                    # Calculate metrics
                    support = count_AB / self.n_transactions
                    confidence = count_AB / count_A
                    lift = (count_AB * self.n_transactions) / (count_A * count_B)

                    # Filter by thresholds
                    if confidence >= min_confidence and lift >= min_lift:
                        rule = {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                        self.association_rules.append(rule)

        # Sort by confidence, then lift
        self.association_rules.sort(
            key=lambda x: (x["confidence"], x["lift"]), reverse=True
        )

        print(f"\nGenerated {len(self.association_rules)} association rules")
        print(f"  Min confidence: {min_confidence}")
        print(f"  Min lift: {min_lift}")

        return self.association_rules

    def get_frequent_itemsets(self, k=None, min_support=None):
        """
        Get frequent itemsets.

        Args:
            k (int): Size of itemsets (None for all)
            min_support (float): Override minimum support filter

        Returns:
            list: Frequent itemsets
        """
        if k is None:
            itemsets = self.all_frequent_itemsets
        else:
            itemsets = self.frequent_itemsets.get(k, [])

        if min_support is not None:
            min_count = int(np.ceil(min_support * self.n_transactions))
            itemsets = [(iset, cnt) for iset, cnt in itemsets if cnt >= min_count]

        return itemsets

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("APRIORI ALGORITHM - RESULTS SUMMARY")
        print("=" * 70 + "\n")

        print(f"Min Support: {self.min_support}")
        print(f"Total Transactions: {self.n_transactions}")
        print(f"Total Frequent Itemsets: {len(self.all_frequent_itemsets)}\n")

        print("Itemsets by Size:")
        for k in sorted(self.frequent_itemsets.keys()):
            count = len(self.frequent_itemsets[k])
            print(f"  {k}-itemsets: {count}")

        if self.association_rules:
            print(f"\nAssociation Rules: {len(self.association_rules)}")
            print("\nTop 10 Rules by Confidence:")
            for i, rule in enumerate(self.association_rules[:10], 1):
                ant_str = ", ".join(sorted(rule["antecedent"]))
                cons_str = ", ".join(sorted(rule["consequent"]))
                print(f"\n  {i}. {{{ant_str}}} -> {{{cons_str}}}")
                print(f"     Support: {rule['support']:.3f}")
                print(f"     Confidence: {rule['confidence']:.3f}")
                print(f"     Lift: {rule['lift']:.2f}")

    def get_rules_df(self):
        """Convert rules to DataFrame for analysis."""
        if not self.association_rules:
            return pd.DataFrame()

        rules_data = []
        for rule in self.association_rules:
            ant_str = ", ".join(sorted(rule["antecedent"]))
            cons_str = ", ".join(sorted(rule["consequent"]))
            rule_str = f"{ant_str} → {cons_str}"

            rules_data.append(
                {
                    "antecedent": rule["antecedent"],
                    "consequent": rule["consequent"],
                    "rule_str": rule_str,
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                }
            )

        return pd.DataFrame(rules_data)
