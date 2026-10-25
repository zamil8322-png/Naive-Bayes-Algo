# Naive Bayes Classifier: Practical Demonstration

## Aim
To demonstrate the Naive Bayes algorithm for classification, implemented from scratch in Python and with scikit-learn.

## Files
- `naive_bayes_dataset.csv`: the small "Play Tennis" dataset (14 rows)
- `naive_base.py`: the implementation code

## Dataset
| Feature     | Values                    |
|-------------|---------------------------|
| Outlook     | Sunny, Overcast, Rain     |
| Temperature | Hot, Mild, Cool           |
| Humidity    | High, Normal              |
| Wind        | Weak, Strong              |
| **Target**  | PlayTennis (Yes / No)     |

## Theory
Naive Bayes is a probabilistic classifier based on Bayes' theorem:

    P(Class | Features) = P(Features | Class) x P(Class) / P(Features)

It assumes that all features are independent of each other given the class (hence "naive"), so:

    P(Class | x1..xn) is proportional to P(Class) x P(x1|Class) x ... x P(xn|Class)

The class with the highest posterior probability is the prediction. Laplace smoothing (+1) is used so that no probability becomes zero.

## Requirements
- Python 3.x
- pandas
- scikit-learn

Install with:

    pip install pandas scikit-learn

## How to Run
1. Put `naive_bayes_dataset.csv` and `naive_base.py` in the same folder.
2. Open a terminal in that folder.
3. Run:

       python naive_base.py

## Steps in the Code
1. Load the dataset with pandas.
2. Calculate the prior probabilities P(Class).
3. Calculate the likelihoods P(Feature | Class) with Laplace smoothing.
4. Predict a new sample by multiplying priors and likelihoods, then normalizing.
5. Calculate the training accuracy.
6. Repeat the prediction using scikit-learn's `CategoricalNB`.

## Output
Test sample: Outlook = Sunny, Temperature = Cool, Humidity = High, Wind = Strong

| Class | Probability |
|-------|-------------|
| Yes   | 0.28        |
| No    | 0.72        |

- **Prediction:** No (do not play tennis)
- **Training accuracy:** 13/14 = 92.86%

## Conclusion
The Naive Bayes classifier predicted the class of the new sample correctly using simple probability calculations. The from-scratch result matches the scikit-learn result, which confirms the implementation.
