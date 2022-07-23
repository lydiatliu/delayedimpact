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

We owe a great deal to Liu et al.'s work, [*Delayed Impact of Fair Machine Learning.*](https://arxiv.org/abs/1803.04383). We extended their [code](https://github.com/lydiatliu/delayedimpact) here to solve a classification problem with 
multiple ML models, fairness metrics, and mitigation methods. 

**Problem Domain**: loan repayment

**Datasets**:
Our simulated datasets are based on Hardt et al.'s dataset: *include link here*

# Project Pipeline

This project can be divided into two stages:
1. Dataset prepping
2. Training and testing ML models
3. Performing statistical analyses on results

This section gives a high-level overview of the workflow of each section and what is needed to run the code.

## 1. Dataset Preparation (*under construction*)

This section prepares the simulated dataset that will be used for training and testing the unmitigated and mitigated models. 
**Key details**:
- See ```....``` for details of how to run this section
- Uses...

## 2. Experimental Pipeline (*under construction*)

**Files for the AC Method**:(*under construction*)
- ```...``` is the pyfile that runs...


<!-- NOTES -->
## Notes (*pare down as I add to the above*):
- Agarwal et al. used weighted classification implementations of logistic regression and gradient-boosted decision trees
- I can figure out the classifier which was picked from the GS classifier graphs by comparing the values.
- Grid search code: https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_grid_search/grid_search.py
- Exponentiated Gradient code: https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py
- Fairness constraint options: DP refers to demographic parity, EO to equalized odds, TPRP to true positive rate parity, FPRP to false positive rate parity, ERP to error rate parity, and BGL to bounded group loss.
- CAN use (these sklearn models' fit functions take in sample weights): gaussian naive bayes, decision tree, logistic regression, and svm
- Couldn't use K-nearest neighbors as an ML classifier bc the fit function does not take in sample weights parameter
- For Fairlearn mitigator algorithms to work, I have to weigh the data. At the moment, I'm weighing all samples equally (weight_index=1) 
- The sklearn confusion matrix looks like:
  ```
  [[TN FP]
   [FN TP]]
  ```
- Inspired by Liu et. al.'s 2018 paper, TPs' scores increase by 75, FPs' scores drop by 150, and TNs and FNs do not change currently. Also, Delayed Impact (DI) is the average score change of each (racial) group.
- About race features: Black is 0 and White it 1.
- Inspired by Liu et. al.'s 2018 paper, TPs' scores increase by 75, FPs' scores drop by 150, and TNs and FNs do not change currently. Also, Delayed Impact (DI) is the average score change of each (racial) group.
- Race features: Black is 0 and White it 1.
- Reduction algorithms and fairness constraints: 'disparity constraints are cast as Lagrange multipliers, which cause the reweighting and relabelling of the input data. This *reduces* the problem back to standard machine learning training.'
- Files to read/understand/change for reduction alg+fairness constraints understandings+additions: fairlearn/fairlearn/reductions/_moments/utility_parity.py, moment.py, __init__.py, ../__init__.py,    

<!-- CONTACT -->
## Contact
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to Lydia for helping me get started using her code!

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
