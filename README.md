# Bayesian Model Averaging

| | Uniform | Poisson | Binomial | Negative Binomial | Beta-Binomial | Empirical | BMA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| # of Parameters | 1 | 1 | 2 | 2 | 2 | 12 | 1.4 |
| BIC | 67 | 69 | 72 | 68 | 69 | 83 | **66** |
| Weights | 41% | 16% | 4% | 25% | 14% | 0% | - |

The Bayesian Average Model is a notable improvement over the individual approaches.

![Bayesian Average Model](output/observed%20data%20vs%20fitted%20models.png)

Cross-Validated BMA with 12 folds provides a minor improvement:

![BMA Cross-Validated Model](output/observed%20data%20vs%20bma%20models%20and%20cv.png)

We can visualize the model uncertainty estimated by cross-validation:

![Model Uncertainty](output/box%20plot%20of%20model%20weights%20of%20components%20in%20bma.png)
