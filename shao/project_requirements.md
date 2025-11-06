# Project 2: Biomedical Data Mining

## Datasets

### Dataset 1: Breast Cancer Wisconsin (Diagnostic)
- **Source**: https://www.kaggle.com/datasets/erdemtaha/cancer-data
- **Records**: 569 patients with breast cancer
- **Features**: 30 numerical features from digitized images of breast mass tissue
- **Target**: Binary classification (M = Malignant, B = Benign)
- **Key Features**:
  - `id`: Unique patient identifier
  - `diagnosis`: Cancer type (M/B)
  - Mean values: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points
  - Additional: standard error and worst-case values for each measurement

### Dataset 2: Disease Symptom Prediction
- **Source**: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- **Structure**:
  - `Disease`: Name of the disease
  - `Symptom_1` to `Symptom_17`: Various symptoms per disease
  - Additional files: symptom descriptions, precautions, severity weights
- **Coverage**: Multiple diseases with symptom patterns and relationships

---

## Task 1: Analysis of Symptom Co-occurrence Patterns (30%)

### Objective
Analyze co-occurrence patterns of symptoms within disease profiles to identify symptom combinations that frequently appear together.

### Method
Implement the **Apriori algorithm** on the Disease Symptom dataset.

### Requirements
1. **Data Preparation**: Each disease = "basket", symptoms = "items"
2. **Analysis**: Identify frequent itemsets of symptoms that co-occur
3. **Data Handling**: May treat as binary (symptom present/absent)
4. **Data Cleaning**:
   - Normalize symptom synonyms (e.g., "fever" vs "pyrexia")
   - Ensure minimum number of transactions
   - May augment with mirrored symptom dataset if needed

### Deliverables
- Document methodology
- Report results of frequent symptom patterns

---

## Task 2: Mining Cancer Feature Patterns (40%)

### Objective
Analyze feature sequences and patterns in cancer diagnosis data to uncover characteristics distinguishing malignant from benign cases through sequential pattern mining.

### Steps

#### 1. Data Preprocessing
- Transform numerical features into categorical sequences
- **Sequence Definition**:
  - Rank features per patient by z-score (or mutual information w.r.t. diagnosis)
  - Group top-k features as ordered itemsets
  - Max sequence length L, maxgap = 1
  - Same-order ties may form single itemset

#### 2. Data Analysis
- Apply **GSP (Generalized Sequential Pattern)** algorithm
- Discover patterns in malignant vs. benign cases

### Example Transformation
```
Patient A: <{high_radius}, {high_texture}, {low_smoothness}>
Patient B: <{low_radius}, {high_compactness}, {high_concavity}>

Discovered patterns:
- <{high_radius}, {high_texture}> → Often malignant
- <{low_radius}, {low_smoothness}> → Often benign
```

### Tips
- **Feature Transformation**: Convert continuous → categorical (low/medium/high) using statistical thresholds
- **Binning Strategies**: Use `KBinsDiscretizer` (uniform / quantile / k-means)
- **Sensitivity Analysis**: Report results across different binning strategies
- Note: 'Sequence' = derived order of discretized features (not temporal)
- Focus on interpretable patterns
- Consider computational complexity

### Deliverables
- Document preprocessing approach
- Report sequential patterns found
- Include sensitivity analysis of binning strategies

---

## Task 3: Open Advanced Tasks (30%)

### Objective
Define and solve a healthcare-related application based on the datasets.

### Possible Applications
- Disease prediction
- Cancer risk assessment
- Symptom-based diagnosis

### Approach
Solution may include (but not limited to):
- Traditional analytics
- Machine learning
- Deep learning
- LLM-related tasks

### Deliverables
- Explain algorithm design
- Document experimental results
- Demonstrate solution effectiveness

---

## General Requirements

### Code
- Write all code yourself
- May learn from online resources but must cite key insights/functions in comments
- Core logic and program structure must be original
- **No code sharing between groups** (screened with MOSS)

### Report (max 8 pages)
- Document methodology for all tasks
- Present results and analysis
- Page limit excludes front page

### Video (max 5 minutes)
- Demo code functionality (screen capture)
- Highlight key report elements