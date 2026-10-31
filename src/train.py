"""Train a spam filter on the official UCI SMS Spam Collection."""
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to SMSSpamCollection")
    args = parser.parse_args()
    df = pd.read_csv(args.data, sep="\t", names=["label", "message"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    spam_score = model.predict_proba(X_test)[:, list(model.classes_).index("spam")]
    print(classification_report(y_test, prediction, digits=3))
    print("confusion_matrix=")
    print(confusion_matrix(y_test, prediction, labels=["ham", "spam"]))
    print(f"spam_average_precision={average_precision_score(y_test == 'spam', spam_score):.3f}")


if __name__ == "__main__":
    main()
