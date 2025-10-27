"""
task1_apriori.py
Task 1: Analysis of Symptom Co-occurrence Patterns using Apriori Algorithm

This module implements the Apriori algorithm to discover frequent symptom
combinations within disease profiles. The algorithm identifies patterns of
symptoms that frequently co-occur in the same disease.

Algorithm Reference:
- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
- Concept: Generate candidate itemsets and prune based on minimum support threshold
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AprioriMiner:
    """
    Apriori algorithm implementation for frequent itemset mining.
    
    This class implements the Apriori algorithm from scratch, discovering
    frequent symptom combinations within disease profiles.
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
        
        print(f"Fitting Apriori with min_support={self.min_support} (count={min_support_count})")
        print(f"Total transactions: {len(transactions)}\n")
        
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
        while len(current_itemsets) > 1 and k < 10:  # Limit to 10-itemsets
            k += 1
            print(f"Generating {k}-itemsets...")
            
            # Generate candidates
            candidates = self._generate_candidates(current_itemsets, k)
            
            if not candidates:
                break
            
            # Prune candidates
            frequent_k_itemsets = self._prune_candidates(candidates, min_support_count)
            
            if not frequent_k_itemsets:
                break
            
            self.frequent_itemsets[k] = frequent_k_itemsets
            current_itemsets = frequent_k_itemsets
            
            print(f"  Found {len(frequent_k_itemsets)} frequent {k}-itemsets\n")
        
        # Collect all frequent itemsets
        for k in sorted(self.frequent_itemsets.keys()):
            self.all_frequent_itemsets.extend(self.frequent_itemsets[k])
        
        print(f"✓ Total frequent itemsets: {len(self.all_frequent_itemsets)}\n")
    
    
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
        
        # Generate candidates by combining (k-1)-itemsets
        candidates = set()
        for i in range(len(itemsets_only)):
            for j in range(i + 1, len(itemsets_only)):
                union = itemsets_only[i] | itemsets_only[j]
                
                # Only keep if union has exactly k items
                if len(union) == k:
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
            For each A ⊂ X:
                If confidence(A → X-A) >= min_confidence:
                    Generate rule A → X-A
        
        Args:
            min_confidence (float): Minimum confidence threshold (0-1)
                                   Confidence = support(A ∪ B) / support(A)
            
        Returns:
            list: List of association rules
        """
        self.association_rules = []
        
        # Create support dictionary for faster lookup
        support_dict = {itemset: support for itemset, support in self.all_frequent_itemsets}
        
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
                            lift = support_AB / (support_A * support_B / len(self.transactions) ** 2)
                        else:
                            lift = 0
                        
                        rule = {
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support_AB / len(self.transactions),
                            'confidence': confidence,
                            'lift': lift
                        }
                        self.association_rules.append(rule)
        
        # Sort by confidence
        self.association_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
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
        print("\n" + "="*70)
        print("APRIORI ALGORITHM - RESULTS SUMMARY")
        print("="*70 + "\n")
        
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
        antecedent = ', '.join(sorted(rule['antecedent']))
        consequent = ', '.join(sorted(rule['consequent']))
        return f"{{{antecedent}}} → {{{consequent}}}"


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_disease_transactions(disease_df, symptom_cols, normalize_symptoms=True):
    """
    Convert disease-symptom data into transaction format for Apriori.
    
    Args:
        disease_df (pd.DataFrame): Disease symptom dataset
        symptom_cols (list): List of symptom column names
        normalize_symptoms (bool): Whether to normalize symptom names
        
    Returns:
        list: List of transactions (frozensets of symptoms)
    """
    transactions = []
    
    for idx, row in disease_df.iterrows():
        # Get symptoms for this disease (non-null values)
        symptoms = set()
        for col in symptom_cols:
            symptom = row[col]
            if pd.notna(symptom) and symptom != '':
                # Normalize: strip whitespace, lowercase
                if normalize_symptoms:
                    symptom = str(symptom).strip().lower().replace(' ', '_')
                else:
                    symptom = str(symptom).strip()
                
                symptoms.add(symptom)
        
        # Only add if disease has at least one symptom
        if symptoms:
            transactions.append(frozenset(symptoms))
    
    return transactions


def normalize_symptom_names(disease_df, symptom_cols):
    """
    Normalize symptom names by identifying and merging synonyms.
    
    Args:
        disease_df (pd.DataFrame): Disease symptom dataset
        symptom_cols (list): List of symptom column names
        
    Returns:
        pd.DataFrame: Dataset with normalized symptom names
    """
    df = disease_df.copy()
    
    # Manual synonym mapping (extend as needed)
    synonym_map = {
        'fever': ['pyrexia', 'high_temperature'],
        'nausea': ['vomiting', 'queasiness'],
        'cough': ['coughing'],
        'headache': ['head_pain'],
        'fatigue': ['tiredness', 'exhaustion', 'weakness'],
    }
    
    # Normalize all symptom columns
    for col in symptom_cols:
        df[col] = df[col].apply(
            lambda x: _normalize_symptom(x, synonym_map) if pd.notna(x) else x
        )
    
    return df


def _normalize_symptom(symptom, synonym_map):
    """
    Normalize a single symptom using synonym mapping.
    
    Args:
        symptom (str): Symptom name
        synonym_map (dict): Mapping of canonical names to synonyms
        
    Returns:
        str: Normalized symptom name
    """
    symptom_normalized = str(symptom).strip().lower().replace(' ', '_')
    
    # Check if it's a synonym
    for canonical, synonyms in synonym_map.items():
        if symptom_normalized in [s.lower().replace(' ', '_') for s in synonyms]:
            return canonical
    
    return symptom_normalized


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_symptom_patterns(miner, top_n=15):
    """
    Analyze and display top symptom patterns.
    
    Args:
        miner (AprioriMiner): Fitted Apriori miner
        top_n (int): Number of top patterns to display
    """
    print("\n" + "="*70)
    print("TOP FREQUENT SYMPTOM COMBINATIONS")
    print("="*70 + "\n")
    
    # Get top itemsets by support
    sorted_itemsets = sorted(miner.all_frequent_itemsets, 
                            key=lambda x: x[1], reverse=True)[:top_n]
    
    for rank, (itemset, support) in enumerate(sorted_itemsets, 1):
        symptoms = ', '.join(sorted(itemset))
        pct = (support / len(miner.transactions)) * 100
        print(f"{rank:2d}. {symptoms:50s} | Support: {support:4d} ({pct:5.1f}%)")


def visualize_apriori_results(miner, output_path='./results'):
    """
    Create visualizations of Apriori results.
    
    Args:
        miner (AprioriMiner): Fitted Apriori miner
        output_path (str): Path to save visualizations
    """
    Path(output_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Task 1: Apriori Algorithm - Symptom Co-occurrence Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution of itemset sizes
    itemset_sizes = defaultdict(int)
    for itemset, _ in miner.all_frequent_itemsets:
        itemset_sizes[len(itemset)] += 1
    
    axes[0, 0].bar(itemset_sizes.keys(), itemset_sizes.values(), 
                   color='#95E1D3', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Distribution of Frequent Itemset Sizes', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Itemset Size')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Top 10 itemsets by support
    top_itemsets = sorted(miner.all_frequent_itemsets, 
                          key=lambda x: x[1], reverse=True)[:10]
    itemset_labels = [', '.join(sorted(list(itemset)[:2]))  # Shorten for display
                     for itemset, _ in top_itemsets]
    supports = [support for _, support in top_itemsets]
    
    axes[0, 1].barh(range(len(supports)), supports, 
                    color='#F38181', edgecolor='black', alpha=0.8)
    axes[0, 1].set_yticks(range(len(supports)))
    axes[0, 1].set_yticklabels(itemset_labels, fontsize=9)
    axes[0, 1].set_title('Top 10 Symptom Combinations by Support', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Support Count')
    axes[0, 1].invert_yaxis()
    
    # 3. Distribution of rule confidence
    if miner.association_rules:
        confidences = [rule['confidence'] for rule in miner.association_rules]
        axes[1, 0].hist(confidences, bins=30, color='#FFB6B9', 
                       edgecolor='black', alpha=0.8)
        axes[1, 0].set_title('Distribution of Rule Confidence', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(confidences), color='red', 
                          linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.2f}')
        axes[1, 0].legend()
    
    # 4. Summary statistics
    summary_text = f"""APRIORI ALGORITHM SUMMARY

Parameters:
  • Min Support: {miner.min_support}
  • Total Transactions: {len(miner.transactions)}

Results:
  • Frequent Itemsets: {len(miner.all_frequent_itemsets)}
  • Association Rules: {len(miner.association_rules)}
  
Itemsets:
  • Max Size: {max([len(itemset) for itemset, _ in miner.all_frequent_itemsets]) if miner.all_frequent_itemsets else 0}
  
Rules:
  • Avg Confidence: {np.mean([r['confidence'] for r in miner.association_rules]):.3f}
  • Avg Lift: {np.mean([r['lift'] for r in miner.association_rules]):.3f}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='#E8F4F8', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/task1_apriori_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}/task1_apriori_results.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for Task 1."""
    import sys
    sys.path.insert(0, '.')
    
    from utils.data_loading import load_datasets
    from utils.preprocessing import clean_disease_data, get_symptom_columns
    
    print("\n" + "="*70)
    print("TASK 1: SYMPTOM CO-OCCURRENCE ANALYSIS - APRIORI ALGORITHM")
    print("="*70 + "\n")
    
    # Load data
    print("Step 1: Loading datasets...")
    _, disease_df = load_datasets()
    disease_df = clean_disease_data(disease_df)
    symptom_cols = get_symptom_columns(disease_df)
    print(f"✓ Loaded {len(disease_df)} disease records with {len(symptom_cols)} symptom columns\n")
    
    # Prepare transactions
    print("Step 2: Preparing transactions...")
    transactions = prepare_disease_transactions(disease_df, symptom_cols)
    print(f"✓ Created {len(transactions)} transactions\n")
    
    # Run Apriori
    print("Step 3: Running Apriori algorithm...")
    miner = AprioriMiner(min_support=0.1)
    miner.fit(transactions)
    
    # Generate rules
    print("Step 4: Generating association rules...")
    miner.generate_rules(min_confidence=0.5)
    
    # Print results
    miner.print_summary()
    analyze_symptom_patterns(miner, top_n=15)
    
    # Visualize
    print("\nStep 5: Creating visualizations...")
    visualize_apriori_results(miner)
    
    print("\n" + "="*70)
    print("✓ TASK 1 COMPLETE")
    print("="*70 + "\n")
    
    return miner


if __name__ == "__main__":
    miner = main()