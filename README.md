\# Privacy-Enhanced Distributed Data Mining (Phase 2)



This repository implements \*\*Phase 2\*\* of a privacy-preserving data mining framework based on \*\*Rotation-Based Data Transformation\*\* and \*\*RSS-RMD (Ratio of Secure Summations using Rational Multiplicative Disturbance)\*\*.



The focus of this phase is the \*\*evaluation of privacy-preserving clustering and classification algorithms\*\* on datasets that have already been transformed using the Phase-1 rotation method.



---



\## Research Context



This work is a continuation of the feature-engineering stage implemented in:



\*\*Phase-1: 4D Rotation Transformation\*\*  

🔗 https://github.com/tanzeelajvd/4d\_rotation\_transformation



In Phase-1, datasets are transformed using a \*\*4D rotation-based transformation\*\* to preserve statistical properties while protecting raw feature values.



In \*\*Phase-2 (this repository)\*\*, the transformed datasets are used for \*\*privacy-preserving distributed learning\*\* using secure aggregation techniques.



---



\## Referenced Research Papers



This implementation follows the methodology described in the following research works:



1\. \*\*A Privacy-Preserving Data Mining Approach Using Rotation-Based Transformation\*\*  

&nbsp;  ScienceDirect (Elsevier)  

&nbsp;  🔗 https://www.sciencedirect.com/science/article/pii/S1319157820304109



2\. \*\*A New Method to Compute Ratio of Secure Summations and Its Application in Privacy-Preserving Distributed Data Mining (RSS-RMD)\*\*  

&nbsp;  IEEE  

&nbsp;  🔗 https://ieeexplore.ieee.org/document/8625413



---



\## Algorithms Implemented



\### 1. Standard K-Means (Baseline)

A conventional K-Means clustering algorithm applied to \*\*rotated datasets\*\* for baseline comparison.



\### 2. RSS-RMD-Based K-Means (Proposed)

A privacy-enhanced K-Means variant where:

\- Cluster centroids are computed using \*\*RSS-RMD\*\*

\- Raw feature values are never directly aggregated

\- Secure ratio of summations replaces the arithmetic mean



---



\### 3. Standard Naive Bayes (Baseline)

A Gaussian Naive Bayes classifier applied to transformed datasets using:

\- Plain class priors

\- Plain feature means and variances



\### 4. RSS-RMD-Based Naive Bayes (Proposed)

A privacy-preserving Naive Bayes classifier where:

\- Class priors and feature means are computed using \*\*RSS-RMD\*\*

\- Only secure ratios of summations are used

\- Raw statistical aggregates are never revealed



This allows evaluation of \*\*classification accuracy under privacy constraints\*\*.



---



\## Repository Structure



```



privacy-enhanced-distributed-data-mining/

│

├── algorithms/

│   ├── \*\*init\*\*.py

│   ├── k\_means\_algo.py        # K-Means with RSS-RMD integration

│   ├── naive\_bayes\_algo.py    # Naive Bayes with RSS-RMD integration

│   └── rss\_algo.py            # RSS and RSS-RMD secure aggregation primitives

│

├── transformed\_datasets/

│   ├── transformed\_bank-full.csv

│   ├── transformed\_Data\_User\_Modeling\_Dataset.csv

│   ├── transformed\_forestfires.csv

│   ├── transformed\_HCV-Egy-Data.csv

│   └── transformed\_seeds\_dataset.csv

│

├── requirements.txt

├── README.md

└── .gitignore



````



---



\## Input Data



All datasets used in this project are \*\*already transformed\*\* using the Phase-1 4D rotation method.



No raw or untransformed datasets are used in this phase.



For classification tasks, the \*\*last column is assumed to be the class label\*\*.



---



\## Experimental Workflow



For each transformed dataset:



1\. Load numerical feature vectors

2\. Partition data across multiple virtual clients

3\. Apply \*\*standard K-Means\*\* and \*\*RSS-RMD-based K-Means\*\*

4\. Apply \*\*standard Naive Bayes\*\* and \*\*RSS-RMD-based Naive Bayes\*\*

5\. Compare:

&nbsp;  - Convergence behavior (for clustering)

&nbsp;  - Execution time

&nbsp;  - Cluster consistency

&nbsp;  - Classification accuracy



---



\## Environment Setup



Install required dependencies using:



```bash

pip install -r requirements.txt

````



---



\## Running the Experiments



\### K-Means Clustering



```bash

python algorithms/k\_means\_algo.py

```



You will be prompted to:



\* Select a transformed dataset

\* Specify the number of data points

\* Choose the number of clusters



---



\### Naive Bayes Classification



```bash

python algorithms/naive\_bayes\_algo.py

```



The script performs:



\* Train–test split

\* Baseline Naive Bayes evaluation

\* RSS-RMD-based Naive Bayes evaluation



---



\## Notes



\* RSS-RMD is used \*\*only for secure aggregation\*\*, not encryption

\* No cryptographic libraries are required

\* PCA or visualization (if used) is for analysis only

\* This implementation prioritizes \*\*algorithmic clarity, scalability, and reproducibility\*\*



---



\## Disclaimer



This project is intended \*\*solely for academic and research purposes\*\*.

It is \*\*not designed for production or real-world deployment\*\*.



---



\## License



Academic and research use only.





