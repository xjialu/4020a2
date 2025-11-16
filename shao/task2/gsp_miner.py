from collections import defaultdict


class GSPMiner:
    def __init__(self, min_support=0.1, max_length=5, max_gap=1):
        """
        Initialize GSP algorithm with itemset support.

        Parameters:
        - min_support: minimum support threshold (as fraction)
        - max_length: maximum sequence length (number of itemsets)
        - max_gap: maximum gap between consecutive itemsets in a sequence

        Sequence format: A sequence is a list of itemsets, where each itemset is a frozenset.
        Example: [frozenset({'high_radius'}), frozenset({'high_texture', 'high_area'})]
        represents <{high_radius}, {high_texture, high_area}>
        """
        self.min_support = min_support
        self.max_length = max_length
        self.max_gap = max_gap

    def get_frequent_1_sequences(self, sequences):
        """
        Find frequent 1-sequences (single items in a single itemset).

        Returns:
        - Dictionary with tuple of frozenset as key: (frozenset({item}),) -> count
        """
        item_counts = defaultdict(int)

        # Count occurrences of each individual item across all sequences
        for sequence in sequences:
            unique_items = set()
            for itemset in sequence:
                unique_items.update(itemset)

            for item in unique_items:
                item_counts[item] += 1

        # Filter by minimum support
        min_count = int(self.min_support * len(sequences))
        frequent_1_seq = {}

        for item, count in item_counts.items():
            if count >= min_count:
                # Store as tuple of frozenset for consistency
                frequent_1_seq[(frozenset({item}),)] = count

        return frequent_1_seq

    def generate_candidates(self, frequent_k_seq):
        """
        Generate candidate (k+1)-sequences from frequent k-sequences.

        Two main operations:
        1. Sequence extension: add a new itemset at the end
        2. Itemset extension: add an item to the last itemset
        """
        candidates = set()
        frequent_seqs = list(frequent_k_seq.keys())

        for i in range(len(frequent_seqs)):
            for j in range(len(frequent_seqs)):
                seq1 = frequent_seqs[i]
                seq2 = frequent_seqs[j]

                # Case 1: Sequence extension (add new itemset)
                # If first k-1 itemsets of seq1 match last k-1 itemsets of seq2
                if len(seq1) >= 1 and len(seq2) >= 1:
                    if seq1[1:] == seq2[:-1]:
                        # Add last itemset of seq2 as new itemset
                        candidate = seq1 + (seq2[-1],)
                        if len(candidate) <= self.max_length:
                            candidates.add(candidate)

                # Case 2: Itemset extension (add item to last itemset)
                # If all but last itemset match, and last itemsets differ by one item
                if len(seq1) >= 1 and len(seq2) >= 1:
                    if seq1[:-1] == seq2[:-1]:
                        # Check if last itemsets differ by exactly one item
                        last1 = seq1[-1]
                        last2 = seq2[-1]

                        # Try to merge last itemsets
                        if last1 != last2:
                            merged = last1 | last2  # Union of frozensets
                            candidate = seq1[:-1] + (merged,)
                            if len(candidate) <= self.max_length:
                                candidates.add(candidate)

        return list(candidates)

    def is_subsequence_iterative(self, subseq, sequence):
        """
        Check if subseq matches sequence with gap constraint.

        Gap constraint: Between consecutive matched itemsets, there can be at most
        max_gap itemsets in the original sequence.

        Note: Uses greedy first-match strategy. Each feature appears at most
        once per sequence (since features are ranked by z-score, each feature
        appears in exactly one itemset per patient sequence).
        """
        if len(subseq) == 0:
            return True
        if len(subseq) > len(sequence):
            return False

        sub_idx = 0
        seq_idx = 0
        last_match_idx = -1

        while sub_idx < len(subseq) and seq_idx < len(sequence):
            pattern_itemset = subseq[sub_idx]

            if pattern_itemset.issubset(sequence[seq_idx]):
                if last_match_idx != -1:
                    gap = seq_idx - last_match_idx - 1
                    if gap > self.max_gap:
                        return False

                last_match_idx = seq_idx
                sub_idx += 1

            seq_idx += 1

        return sub_idx == len(subseq)

    def count_support(self, candidate, sequences):
        """
        Count support for a candidate sequence.

        Parameters:
        - candidate: tuple of frozensets
        - sequences: list of sequences (each sequence is a list of frozensets)

        Returns:
        - count: number of sequences containing the candidate
        """
        count = 0
        for sequence in sequences:
            if self.is_subsequence_iterative(candidate, sequence):
                count += 1
        return count

    def mine_patterns(self, sequences):
        """
        Mine frequent sequential patterns using GSP algorithm.

        Parameters:
        - sequences: list of sequences, where each sequence is a list of frozensets
                    Example: [[frozenset({'A'}), frozenset({'B', 'C'})],
                             [frozenset({'A'}), frozenset({'D'})]]

        Returns:
        - frequent_patterns: dictionary of frequent patterns and their support count
                           Key format: tuple of frozensets
        """
        # Find frequent 1-sequences
        frequent_patterns = {}
        frequent_k_seq = self.get_frequent_1_sequences(sequences)
        frequent_patterns.update(frequent_k_seq)

        k = 1
        min_count = int(self.min_support * len(sequences))

        while frequent_k_seq and k < self.max_length:
            print(
                f"Mining {k + 1}-sequences... Found {len(frequent_k_seq)} frequent {k}-sequences"
            )

            # Generate candidates
            candidates = self.generate_candidates(frequent_k_seq)
            print(f"Generated {len(candidates)} candidates for {k + 1}-sequences")

            # Count support for candidates
            frequent_k_plus_1_seq = {}
            for candidate in candidates:
                support_count = self.count_support(candidate, sequences)
                if support_count >= min_count:
                    frequent_k_plus_1_seq[candidate] = support_count

            frequent_patterns.update(frequent_k_plus_1_seq)
            frequent_k_seq = frequent_k_plus_1_seq
            k += 1

        return frequent_patterns
