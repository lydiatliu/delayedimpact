# delayedimpact

This repository is forked and adapted from Lydia T. Liu's [repository](https://github.com/lydiatliu/delayedimpact) that contains code for reproducing experiment results in:
**Lydia T. Liu, Sarah Dean, Esther Rolf, Max Simchowitz, Moritz Hardt.** [*Delayed Impact of Fair Machine Learning.*](https://arxiv.org/abs/1803.04383) Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018.

This repository is **under construction! :D**

## Todos:
- Update rest of nb code for reflecting the dict results methods
- In results sheet, include up or down arrow to show whether we want large or small values
- Make nb suitable for running all (if possible?) algorithms
- Get results for all
- Try AUC(?) and gradient boosted trees from scikit learn to compare to agarwal's reduction alg paper
- Make notebooks for the different models that show graphs of the GS models with constraint by accuracy, so you can see the different model results
- Try SVM with fewer data points, see what its limit is with size of data? maybe leave 10k run going for a while? don't forget to save results
- Add more fairness constraints to the backend

## Notes:
- For my data, simData_oom10 has 10k rows, simData_oom50 has 50k rows...
- For the Grid search classifier graphs, maybe make balanced accuracy --> accuracy? Or keep them? 
- I can figure out the classifier which was picked from the GS classifier graphs by comparing the values.
- Grid search code: https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_grid_search/grid_search.py
- Exponentiated Gradient code: https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py
- Fairness constraint options: DP refers to demographic parity, EO to equalized odds, TPRP to true positive rate parity, FPRP to false positive rate parity, ERP to error rate parity, and BGL to bounded group loss.
- CAN use (these sklearn models' fit functions take in sample weights): gaussian naive bayes, decision tree, logistic regression, and svm
- Couldn't use K-nearest neighbors as an ML classifier bc the fit function does not take in sample weights parameter
- For Fairlearn mitigator algorithms to work, I have to weigh the data
- ^At the moment, I'm weighing all samples equally (weight_index=1) 
- The sklearn confusion matrix looks like:
  ```
  [[TN FP]
   [FN TP]]
  ```
- Inspired by Liu et. al.'s 2018 paper, TPs' scores increase by 75, FPs' scores drop by 150, and TNs and FNs do not change currently. Also, Delayed Impact (DI) is the average score change of each (racial) group.
- About race features: Black is 0 and White it 1.
- About the reduction algorithms and fairness constraints: 'disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.'
- Files to read/understand/change for reduction alg+fairness constraints understandings+additions: fairlearn/fairlearn/reductions/_moments/utility_parity.py, moment.py, __init__.py, ../__init__.py,    
- Bounded group loss metric not working, need a loss parameter. Definition: 'asks that the prediction error restricted to any protected group remain below some pre-determined level' from https://arxiv.org/abs/1905.12843

## Questions:
- Is the DI calculated with the whole racial group in mind or the subgroup of the racial group who was given a loan (positive classes only)?
- Do I need to scale the data?? This standardizes features by removing mean and scaling to unit variance. 
``` 
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 
```


## Acknowledgments
Thank you to Lydia for helping me get started using her code!

## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
