import pandas as pd

# ---------- Step 1: Load the dataset ----------
df = pd.read_csv("naive_bayes_dataset.csv")
print(df, "\n")

target = "PlayTennis"
features = [c for c in df.columns if c != target]


# ---------- Step 2: Naive Bayes FROM SCRATCH ----------
def train(df):
    """Calculate priors P(class) and likelihoods P(feature=value | class)."""
    priors = df[target].value_counts(normalize=True).to_dict()
    likelihoods = {}
    for cls in priors:
        subset = df[df[target] == cls]
        likelihoods[cls] = {}
        for f in features:
            n_values = df[f].nunique()
            counts = subset[f].value_counts()
            # Laplace smoothing (+1) avoids zero probabilities
            likelihoods[cls][f] = {
                v: (counts.get(v, 0) + 1) / (len(subset) + n_values)
                for v in df[f].unique()
            }
    return priors, likelihoods


def predict(sample, priors, likelihoods):
    """Posterior is proportional to P(class) * product of P(feature | class)."""
    scores = {}
    for cls in priors:
        p = priors[cls]
        for f, v in sample.items():
            p *= likelihoods[cls][f][v]
        scores[cls] = p
    total = sum(scores.values())
    probs = {c: s / total for c, s in scores.items()}  # normalize
    return max(probs, key=probs.get), probs


priors, likelihoods = train(df)
print("Priors:", priors, "\n")

new_day = {"Outlook": "Sunny", "Temperature": "Cool",
           "Humidity": "High", "Wind": "Strong"}
label, probs = predict(new_day, priors, likelihoods)
print("From scratch ->", new_day)
print("Prediction:", label)
print("Probabilities:", {k: round(v, 3) for k, v in probs.items()}, "\n")

# Accuracy on the training data
correct = sum(
    predict(row[features].to_dict(), priors, likelihoods)[0] == row[target]
    for _, row in df.iterrows()
)
print(f"Training accuracy: {correct}/{len(df)} = {correct / len(df):.2%}\n")


# ---------- Step 3: Same thing with scikit-learn ----------
try:
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    enc = OrdinalEncoder()
    X = enc.fit_transform(df[features])
    y = LabelEncoder().fit(df[target])
    model = CategoricalNB(alpha=1.0)  # alpha=1 is Laplace smoothing
    model.fit(X, y.transform(df[target]))

    x_new = enc.transform(pd.DataFrame([new_day]))
    pred = model.predict(x_new)[0]
    proba = model.predict_proba(x_new)[0]
    print("scikit-learn prediction:", y.inverse_transform([pred])[0])
    print("Probabilities:", dict(zip(y.classes_, proba.round(3))))
except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")