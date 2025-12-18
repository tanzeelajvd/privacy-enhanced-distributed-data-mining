# Privacy-Enhanced Distributed Data Mining (Phase 2)

This repository implements **Phase 2** of a privacy-preserving distributed data mining framework based on **Rotation-Based Data Transformation** and **RSS-RMD (Ratio of Secure Summations using Rational Multiplicative Disturbance)**.

The focus of this phase is the **evaluation of privacy-preserving clustering and classification algorithms** on datasets that have already been transformed using the Phase-1 rotation-based method.

---

## Research Context

This work is a continuation of the feature engineering stage implemented in:

**Phase-1: 4D Rotation Transformation**  
🔗 https://github.com/tanzeelajvd/4d_rotation_transformation

In Phase-1, datasets are transformed using a **4D rotation-based transformation** that preserves statistical and geometric properties while protecting raw feature values.

In **Phase-2 (this repository)**, the transformed datasets are used for **privacy-preserving distributed learning** using secure aggregation techniques without revealing sensitive statistical aggregates.

---

## Referenced Research Papers

This implementation follows the methodology described in the following research works:

1. **A Privacy-Preserving Data Mining Approach Using Rotation-Based Transformation**  
   *ScienceDirect (Elsevier)*  
   🔗 https://www.sciencedirect.com/science/article/pii/S1319157820304109

2. **A New Method to Compute Ratio of Secure Summations and Its Application in Privacy-Preserving Distributed Data Mining (RSS-RMD)**  
   *IEEE*  
   🔗 https://ieeexplore.ieee.org/document/8625413

---

## Algorithms Implemented

### 1. Standard K-Means (Baseline)
A conventional K-Means clustering algorithm applied to **rotated datasets** to serve as a baseline for comparison.

### 2. RSS-RMD-Based K-Means (Proposed)
A privacy-enhanced K-Means variant in which:
- Cluster centroids are computed using **RSS-RMD**
- Raw feature values are never directly aggregated
- Secure ratios of summations replace the arithmetic mean

This preserves clustering behavior while enforcing privacy constraints.

---

### 3. Standard Naive Bayes (Baseline)
A Gaussian Naive Bayes classifier applied to transformed datasets using:
- Plain class priors
- Plain feature means and variances

### 4. RSS-RMD-Based Naive Bayes (Proposed)
A privacy-preserving Naive Bayes classifier in which:
- Class priors and feature means are computed using **RSS-RMD**
- Only secure ratios of summations are used
- Raw statistical aggregates are never revealed

This enables evaluation of **classification performance under privacy constraints**.

---

## Repository Structure

```

privacy-enhanced-distributed-data-mining/
│
├── algorithms/
│   ├── **init**.py
│   ├── k_means_algo.py        # K-Means with RSS-RMD integration
│   ├── naive_bayes_algo.py    # Naive Bayes with RSS-RMD integration
│   └── rss_algo.py            # RSS and RSS-RMD secure aggregation primitives
│
├── transformed_datasets/
│   ├── transformed_bank-full.csv
│   ├── transformed_Data_User_Modeling_Dataset.csv
│   ├── transformed_forestfires.csv
│   ├── transformed_HCV-Egy-Data.csv
│   └── transformed_seeds_dataset.csv
│
├── results/
│   ├── bank_dataset.png
│   ├── forestfires.png
│   └── modelling_dataset.png
│
├── requirements.txt
├── README.md
└── .gitignore

````

---

## Input Data

All datasets used in this project are **already transformed** using the Phase-1 4D rotation-based method.

No raw or untransformed datasets are used in this phase.

For classification experiments, the **last column of each dataset is assumed to be the class label**.

---

## Experimental Workflow

For each transformed dataset:

1. Load numerical feature vectors
2. Partition data across multiple virtual clients
3. Apply **standard K-Means** and **RSS-RMD-based K-Means**
4. Apply **standard Naive Bayes** and **RSS-RMD-based Naive Bayes**
5. Compare:
   - Convergence behavior (for clustering)
   - Execution time
   - Cluster consistency
   - Classification accuracy
   - Stability under varying training sizes

---

## Results

The `results/` directory contains saved figures generated during experimentation, including:

- Silhouette-based validation curves for K-Means
- Accuracy validation curves for Naive Bayes
- Dataset-specific comparison plots

These figures are provided for reproducibility and qualitative analysis.

---

## Environment Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
````

---

## Running the Experiments

### K-Means Clustering

```bash
python -m algorithms.k_means_algo
```

You will be prompted to:

* Select a transformed dataset
* Specify the number of data points
* Choose the number of clusters

The script also generates a **silhouette-based validation curve** comparing standard K-Means and RSS-RMD K-Means.

---

### Naive Bayes Classification

```bash
python -m algorithms.naive_bayes_algo
```

The script performs:

* Train–test split
* Baseline Naive Bayes evaluation
* RSS-RMD-based Naive Bayes evaluation
* Validation curve across different training sizes

---

## Notes

* RSS-RMD is used **only for secure aggregation**, not encryption
* No cryptographic libraries are required
* Visualization is used solely for analysis and comparison
* This implementation prioritizes **algorithmic clarity, stability, and reproducibility**

---

## Disclaimer

This project is intended **solely for academic and research purposes**.
It is **not designed for production or real-world deployment**.

---

## License

Academic and research use only.
