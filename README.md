# delayedimpact

This repository is forked and adapted from Lydia T. Liu's [repository](https://github.com/lydiatliu/delayedimpact) that contains code for reproducing experiment results in:
**Lydia T. Liu, Sarah Dean, Esther Rolf, Max Simchowitz, Moritz Hardt.** [*Delayed Impact of Fair Machine Learning.*](https://arxiv.org/abs/1803.04383) Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018.

This repository is **under construction! :D**

## Todos:
- Check what I should put first for predict function model or the instantiated classifier itself? Or should I just take out the model = part all together??
- Add the delayed impact results to the evaluation of results
- Calculate sample weights and get results, [tips](http://www.surveystar.com/startips/weighting.pdf), [sample weight design effects](https://www.nlsinfo.org/content/cohorts/nlsy97/using-and-understanding-the-data/sample-weights-design-effects/page/0/0/#intro)
- Try to figure out bounded group loss metric, need a loss parameter. Definition: 'asks that the prediction error restricted to any protected group remain below some pre-determined level' from https://arxiv.org/abs/1905.12843
- See if I can get two conf matrices to be printed (black and white groups) in one cell's output

## Notes:
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
