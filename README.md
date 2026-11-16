Domain: E-commerce

## How to run

Run these commands from the repository root using Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The data generator must print `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`. It creates the synthetic `data/records.csv`. Each Part script reads that file and regenerates its results in `outputs/`. Only Python's standard library and NumPy are used. All reported calculations are performed by the submitted code.

## Part 1 findings

The estimated mean order value is **8.9382 hundreds of INR**, with a 95% confidence interval of **8.3026 to 9.5738**. The test statistic is **z = 2.8930**, which exceeds the two-sided critical value 1.96, so the null hypothesis that the population mean is 8.0 (INR 800) is rejected at the 5% level. A Type I error would mean concluding that the true average order value differs from INR 800 when it actually equals INR 800. The estimated gap is **0.9382 hundreds of INR (INR 93.82)**, which is large enough to be worth reviewing in planning, although statistical significance alone does not establish business importance.

The required sample size for a margin of error of INR 50 is **1,293**, compared with the **800** available records. The current sample is therefore 493 records short of the calculated target. The gradient-descent slope gap is **0.0000** to four decimal places: both methods minimise the same squared-error objective, and the standardised gradient-descent solution converts back to the closed-form line.

The raw residual standard-deviation ratio is **3.7525**, indicating a clear funnel because residual spread is much larger in the upper fitted-value half. The log-log ratio is **1.0396**, so the log transformation largely removes that pattern. In the multiple regression, the coefficients whose absolute values exceed 0.05 are **X2 (complaint count), X3 (discount-order share), and X5 (loyalty score)**. The adjusted R² is **0.7403** for the multiple regression versus **0.6561** for the log-log model, so the additional predictors improve in-sample fit under the required comparison. Coefficients have different units and should not be treated as a direct ranking of importance.

## Part 2 findings

After the required stratified split there are **560 training rows (119 positives)** and **240 test rows (51 positives)**. The 200-update losses, ordered from lowest to highest, are **0.3703 (η = 1.0), 0.3776 (η = 0.1), and 0.5091 (η = 0.01)**. The small learning rate makes slow progress over 200 updates; η = 1.0 performs best among the three tested rates on this dataset and run length.

The three largest absolute logistic-regression weights are **X3 (0.9013), X1 (-0.5554), and X2 (0.5505)**. X3 has the largest odds ratio, **2.4629**, meaning that a one-training-standard-deviation increase in discount-order share multiplies the modelled churn odds by about 2.46 when the other predictors are held fixed.

The always-negative baseline has accuracy **0.7875**, while logistic regression at τ = 0.5 has accuracy **0.8042**. Accuracy alone is weak here because only 51 of 240 test records are positive, so a model can appear accurate while missing many churners. Lowering τ from 0.5 to 0.3 raises recall from **0.4706 to 0.7451**, while precision falls from **0.5455 to 0.4634**.

At τ = 0.3, logistic regression has **F1 = 0.5714** and **recall = 0.7451**. KNN has **F1 = 0.3696** and **recall = 0.3333**. The cost-selected logistic threshold is **τ = 0.15**: its test cost is **INR 6,050**, compared with **INR 11,800** at τ = 0.5 and **INR 20,400** when nobody is flagged. I would deploy the logistic model initially with τ = 0.15, subject to validation on genuine customer data. The threshold is selected using training rows, not test rows, to avoid using the held-out evaluation set to tune the model.

## Part 3 findings

The four successive WCSS drops are **659.8282** (K2 to K3), **108.0739** (K3 to K4), **171.0575** (K4 to K5), and **45.5466** (K5 to K6). The elbow rule selects **K = 3**, because the K3-to-K4 drop is less than half the preceding K2-to-K3 drop. The drops do not shrink steadily; the brief's pinned starting rows provide a reproducible solution, not necessarily the global optimum.

| Segment | Profile from the generated data | Suggested customer action | Churn outcome rate |
|---|---|---|---:|
| A | Long-tenure, low-complaint, low-discount shoppers with higher loyalty scores | Reward loyalty and recommend relevant complementary products | 0.0186 |
| B | Mid-tenure shoppers with high complaint counts and the highest average order value | Resolve service friction quickly and provide proactive order support | 0.2353 |
| C | Short-tenure, discount-heavy shoppers with lower loyalty scores | Improve onboarding and test targeted retention offers | 0.4664 |

Segment **C** has the highest outcome rate at **0.4664** and is the clearest retention priority in this synthetic dataset.

At the dendrogram cut, single linkage gives four clusters: **{E1}, {E2, E3, E6}, {E4}, {E5}**. Complete linkage gives two clusters: **{E1, E4, E5}** and **{E2, E3, E6}**. Single linkage merges clusters when their closest pair is near, which can create a chain of linked observations. Complete linkage considers the farthest cross-cluster pair, so it is more conservative and delays the final merge on this table.

The standardised five-feature matrix has a mean vector of zero, up to floating-point rounding. PC1 is driven most strongly by **X1 (+0.5317), X3 (-0.5187), X5 (+0.4418), and X2 (-0.4100)**, with X4 contributing **-0.2916**. Three principal components are needed to retain at least 80% of the variance (**86.16%**). Mean PC1 scores are **1.7773 for A**, **-0.7831 for B**, and **-1.6709 for C**, showing that A is separated from B and especially C along the first component.

## Part 4 findings

**S4 (customer tenure X1 ≤ 24 months)** is the best split by both information gain (**0.1252**) and Gini gain (**0.0606**). It is best because it produces the largest reduction in class impurity among the four tested rules. Growing a tree until every leaf is pure would fit noise and individual records in this small training set; a maximum-depth limit or minimum-leaf-size rule can stop excessive growth.

For the five test-set models in the voting comparison, logistic regression has the strongest F1 (**0.5714**) while the stump has slightly higher recall (**0.7647**) but lower F1 (**0.5532**). Hard voting has F1 **0.5344** and recall **0.6863**; soft voting has F1 **0.5507** and recall **0.7451**. Combining these models did not improve F1 over logistic regression alone. The two misclassified points after round-one AdaBoost normalisation (points 3 and 7) carry total weight **0.5000**. Round two gives h2 more influence because its weighted error is **0.1667**, lower than round one's **0.2500**, producing a larger alpha (**0.8047** versus **0.5493**).

For Model A option 2, the image input is 28 × 28 × 1. Convolution and pooling output shapes are calculated using the formulas in the brief:

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv 1 (5 × 5, 6 filters, padding 2) | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| Max-pool 1 (2 × 2, stride 2) | 14 × 14 × 6 | 0 |
| Conv 2 (5 × 5, 16 filters, no padding) | 10 × 10 × 16 | 2,416 |
| ReLU | 10 × 10 × 16 | 0 |
| Max-pool 2 (2 × 2, stride 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense 1 + ReLU | 32 | 12,832 |
| Dense output | 10 classes | 330 |

The fully connected alternative has **13,002** parameters, while the CNN has **15,734** parameters for the specified 28 × 28 task. For a 128 × 128 × 3 input and 1,024 fully connected neurons, the fully connected layer has **50,332,672** parameters; a 3 × 3 convolution with 64 filters has **1,792** parameters. Convolution shares each filter's weights across image positions, which is why its parameter count is far smaller.

For Model B, the recurrent parameter shapes are **W_xh: 128 × 64**, **W_hh: 128 × 128**, **W_hy: 16 × 128**, **b_h: 128**, and **b_y: 16**, for **26,768** parameters. The total does not grow with sequence length because the same recurrent weights are reused at every time step. For 10 image classes, I would use ReLU in hidden layers and softmax with categorical cross-entropy at the output. A very deep plain network can suffer vanishing gradients, so early layers receive tiny learning signals; skip connections provide a shorter path for information and gradients to flow through the network.

## Model-choice memo

To: Head of the Retention Team

I recommend the logistic-regression model for an initial structured-data retention pilot, using the training-selected alert threshold of **0.15**. On the held-out test set, its cost at that threshold is **INR 6,050**, compared with **INR 11,800** at 0.5 and **INR 20,400** when no customers are flagged. At τ = 0.3, logistic regression records **F1 = 0.5714** and **recall = 0.7451**, ahead of KNN's **F1 = 0.3696** and **recall = 0.3333**. Neither hard voting (**F1 = 0.5344**) nor soft voting (**F1 = 0.5507**) improves on logistic-regression F1. The evidence is limited because the 800 records are synthetic and the test set contains only 51 positive cases. I would not build Model A or Model B before validating the structured model on representative real customer data; the neural-network work here is only a parameter-sizing exercise.

## Limitations

- The 800 customer records are synthetic and do not represent actual Flipkart customers, products, policies or performance.
- Results depend on one prescribed train/test split and may vary under another sample.
- The test set contains only 51 churn-positive records, so metric estimates can be unstable.
- The model uses seven structured predictors and omits seasonality, product mix, marketing exposure and other possible drivers.
- The false-alarm cost of INR 50 and missed-churn cost of INR 400 are scenario assumptions, not measured business costs.
