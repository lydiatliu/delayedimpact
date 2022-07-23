# Supposedly Fair Classification Systems and Their Impacts
Mackenzie Jorgensen, Elizabeth Black, Natalia Criado, & Jose Such

The algorithmic fairness field has boomed with discrimination mitigation methods to make Machine Learning (ML) model
predictions fairer across individuals and groups. However, recent research shows that these measures can sometimes lead
to harming the very people Artificial Intelligence practitioners want to uplift. In this paper, we take this research a step
further by including real ML models, multiple fairness metrics, and discrimination mitigation methods in our experiments to
understand their relationship with the impact on groups being classified. We highlight how carefully selecting a fairness
metric is not enough when taking into consideration later effects of a model’s predictions–the ML model, discrimination
mitigation method, and domain must be taken into account. Our experiments show that most of the mitigation methods,
although they produce “fairer” predictions, actually do not improve the impact for the disadvantaged group, and for those
methods that do improve impact, the improvement is minimal. We highlight that using mitigation methods to make models
more “fair” can have unintended negative consequences, particularly on groups that are already disadvantaged.

We owe a great deal to Liu et al.'s work, [*Delayed Impact of Fair Machine Learning*](https://arxiv.org/abs/1803.04383). We extended their [code](https://github.com/lydiatliu/delayedimpact) here to solve a classification problem with 
multiple ML models, fairness metrics, and mitigation methods. 

**Problem Domain**: loan repayment

**Datasets**:
Our simulated datasets are based on Hardt et al.'s 2016 dataset. 
- Download the data folder from the Github repository for [fairmlbook](https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore) (Barocas, Hardt and Narayanan 2018)
- Save it to the root directory of this repository
- Then run: ```delayedimpact/notebooks/FICO-figures.ipynb```

# Project Pipeline

This project can be divided into two stages:
1. Dataset prepping
2. Training and testing ML models
3. Performing statistical analyses on results (under construction)

This section gives a high-level overview of the workflow of each section and what is needed to run the code.

## 1. Dataset Preparation

This section prepares the simulated dataset that will be used for training and testing the unmitigated and mitigated models. 

**Key details**:
- The notebook that creates the simulated datasets is: ```delayedimpact/notebooks/simData_collection```
- As of now, you will have to manually input the name of the csvs you want to output
- Relevant parameters to specify: round_num, sample sizes/ratio, name of csvs, order of magnitude
- Future work on this notebook will make the running of this smoother with a cleaner way to set variables and a run file

## 2. Training and Testing ML Models

This section trains ML models on the simulated data and trains unmitigated and mitigated models on it for comparison. 

**Key details**:
- The ```delayedimpact/notebooks/impt_functions.py``` is the pyfile that includes all of the helpful functions for the notebooks
- The ```delayedimpact/notebooks/simData_classification.ipynb``` is the notebook that will train an unmitigated and mitigated ML models
- Relevant parameters to specify: model name, data file, and the boolean save. These are all at the top of the notebook for ease of use
- Future work on this notebook will make the running of this smoother with a cleaner way to set variables and a run file

## 3. Performing statistical analyses on results

This work is under construction.

<!-- NOTES -->
## Notes/Resources:
- For the reduction algorithm code see: [Grid Search](https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_grid_search/grid_search.py) and [Exponentiated Gradient](https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py)
- Reduction algorithms and fairness constraints: 'disparity constraints are cast as Lagrange multipliers, which cause the reweighting and relabelling of the input data. This *reduces* the problem back to standard machine learning training.'
- Fairness constraint options: DP refers to demographic parity, EO to equalized odds, TPRP to true positive rate parity, FPRP to false positive rate parity, ERP to error rate parity, and BGL to bounded group loss.
- The ML models available (these sklearn models' fit functions take in sample weights which is necessary for Fairlearn): gaussian naive bayes, decision tree, logistic regression, and svm. Currently, all samples equally (weight_index=1).
- The sklearn confusion matrix looks like:
  ```
  [[TN FP]
   [FN TP]]
  ```
- Impact score changes: TPs' scores increase by 75, FPs' scores drop by 150, and TNs and FNs do not change currently. Also, for aggregate analyses, we use the average score change of each (racial) group.
- Race features: Black is 0 and White it 1.   

<!-- CONTACT -->
## Contact
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to Lydia for helping me get started using her code!

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
