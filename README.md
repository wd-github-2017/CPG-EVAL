# CPG-EVAL

This repository contains the dataset and evaluation code for the paper:

**CPG-EVAL: Evaluating the Readiness of Large Language Models as Assistants and Teammates in Language Teaching**

## 📂 Repository Contents

This repository provides the core data and analysis scripts used in the study:

*   **`CPG-EVAL-core&results.csv`**
    *   This dataset contains the **CPG-EVAL-core** question bank.
    *   It includes the detailed response results from all subjects, including various Large Language Models (LLMs) and human participants.

*   **`boxplot&group_ranking.py`**
    *   This script performs the statistical analysis and visualization:
        1.  **Visualization**: Generates boxplots to illustrate accuracy distributions across different tasks and groups.
        2.  **Statistical Testing**: Conducts **mixed-effects model** analyses (using GLMM and LMM) to verify the statistical significance of the Group rankings.

## 🚀 Usage

### Dependencies

To run the analysis script, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib scipy statsmodels openpyxl
```

### Running the Analysis

You can reproduce the statistical tests and plots by running:

```bash
python "boxplot&group_ranking.py"
```

## 📖 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@Article{informatics13020029,
AUTHOR = {Wang, Dong},
TITLE = {CPG-EVAL: Evaluating the Readiness of Large Language Models as Assistants and Teammates in Language Teaching},
JOURNAL = {Informatics},
VOLUME = {13},
YEAR = {2026},
NUMBER = {2},
ARTICLE-NUMBER = {29},
URL = {https://www.mdpi.com/2227-9709/13/2/29},
ISSN = {2227-9709},
ABSTRACT = {Large language models (LLMs) have begun to function as assistants or teammates in language learning, teaching, and research. However, what prerequisites are required for LLMs to reliably play these roles, and how such prerequisites should be measured, remains under-discussed. This study focuses on measuring Pedagogical Grammar Pattern Recognition (P-GPR) and establishes the Chinese Pedagogical Grammar Evaluation (CPG-EVAL), a multi-tiered benchmark designed to evaluate P-GPR within International Chinese Language Education. CPG-EVAL operationalizes grammar–instance correspondence through five task types that progressively increase contextual load and interference. We evaluate multiple proprietary and open-source LLMs as well as human participants. Results show a monotonic ordering across groups (humans > larger-scale models > semi-larger-scale models > smaller-scale models). In comparison with human participants, LLM performance is more sensitive to task-format complexity. In addition, we identify a set of completely failed items that consistently mislead all evaluated LLMs, exposing shared and systematic weaknesses in current models’ pedagogical grammar recognition. Overall, this study provides an operational framework for diagnosing the capabilities and risks of LLMs when they are deployed as assistants or teammates in grammar-related language-education tasks and offers empirical reference for safer and more syllabus-aligned use of LLMs in educational settings.},
DOI = {10.3390/informatics13020029}
}
```
