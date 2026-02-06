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
@article{cpg_eval_2026,
  title={CPG-EVAL: Evaluating the Readiness of Large Language Models as Assistants and Teammates in Language Teaching},
  author={Dong Wang},
  journal={Informatics},
  year={2026}
}
```
