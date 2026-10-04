"""Leave-one-condition-out eval: confusion matrix + per-condition F1."""
from sklearn.metrics import confusion_matrix, f1_score


def report(y_true, y_pred, conditions):
    print("Overall F1:", f1_score(y_true, y_pred, average="macro"))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))
    for c in sorted(set(conditions)):
        idx = [i for i, x in enumerate(conditions) if x == c]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        print(f"{c}: F1={f1_score(yt, yp, average='macro'):.3f} n={len(idx)}")
