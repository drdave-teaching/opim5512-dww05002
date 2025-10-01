
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# Find repo root
REPO = Path.cwd()
for _ in range(10):
    if (REPO / ".git").exists():
        break
    if REPO.parent == REPO:
        break
    REPO = REPO.parent

A02 = REPO / "A02-collab"
DATA = A02 / "data"
FIGS = A02 / "figs"
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def cmd_load():
    cal = fetch_california_housing(as_frame=True)
    df = cal.frame.rename(columns={"MedHouseVal": "MedHomeVal_100k"})
    df["MedHomeVal_$"] = df["MedHomeVal_100k"] * 100_000
    df.to_csv(DATA / "raw.csv", index=False)
    print("Saved:", DATA / "raw.csv", "| shape:", df.shape)

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop NA rows (dataset typically has none)
    df = df.dropna(axis=0).copy()
    # Normalize column names (already snake_case, but keep pattern)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def cmd_clean():
    df = pd.read_csv(DATA / "raw.csv")
    df = _clean_df(df)
    df.to_csv(DATA / "clean.csv", index=False)
    print("Saved:", DATA / "clean.csv", "| shape:", df.shape)

def _train_tree(df: pd.DataFrame, target="MedHomeVal_$", max_depth=None, random_state=42):
    y = df[target]
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    r2 = r2_score(y_test, yhat)
    rmse = mean_squared_error(y_test, yhat, squared=False)
    return model, X_test, y_test, yhat, {"r2": float(r2), "rmse": float(rmse), "max_depth": max_depth}

def cmd_model(max_depth=None):
    df = pd.read_csv(DATA / "clean.csv")
    model, X_test, y_test, yhat, metrics = _train_tree(df, max_depth=max_depth)
    # Save metrics & preds
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"y": y_test.to_numpy(), "yhat": yhat}).to_csv(DATA / "preds_test.csv", index=False)
    print("Saved:", DATA / "metrics.json", "and preds_test.csv")

    # Residuals plot
    plt.figure()
    plt.scatter(yhat, y_test - yhat, s=5, alpha=0.5)
    plt.axhline(0, lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - ŷ)")
    plt.tight_layout()
    plt.savefig(FIGS / "residuals.png", dpi=150)

    # Pred vs Actual
    plt.figure()
    plt.scatter(yhat, y_test, s=5, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGS / "pred_vs_actual.png", dpi=150)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=df.drop(columns=["MedHomeVal_$"]).columns)
    importances.sort_values(ascending=False).head(12).plot(kind="bar")
    plt.tight_layout()
    plt.savefig(FIGS / "feat_importance.png", dpi=150)

    # Optional: tree preview if shallow
    if model.get_depth() <= 4:
        plt.figure(figsize=(10, 6))
        plot_tree(model, feature_names=df.drop(columns=["MedHomeVal_$"]).columns, filled=False, impurity=False, max_depth=3)
        plt.tight_layout()
        plt.savefig(FIGS / "tree_preview.png", dpi=150)

def cmd_evaluate():
    preds = pd.read_csv(DATA / "preds_test.csv")
    if preds.empty:
        raise SystemExit("Run 'python src/pipeline.py model' first.")
    preds["decile"] = pd.qcut(preds["yhat"], 10, labels=False, duplicates="drop")
    err = preds.groupby("decile").apply(lambda g: np.sqrt(((g["y"]-g["yhat"])**2).mean())).rename("rmse").reset_index()
    err.to_csv(DATA / "error_by_decile.csv", index=False)
    # Plot
    plt.figure()
    plt.plot(err["decile"], err["rmse"], marker="o")
    plt.xlabel("Predicted decile (low→high)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(FIGS / "error_by_decile.png", dpi=150)
    print("Saved:", DATA / "error_by_decile.csv", "and", FIGS / "error_by_decile.png")

if __name__ == "__main__":
    import sys
    import argparse
    ap = argparse.ArgumentParser(description="A02 Decision Tree pipeline")
    ap.add_argument("cmd", choices=["load", "clean", "model", "evaluate"], help="which step to run")
    ap.add_argument("--max_depth", type=int, default=None, help="decision tree max_depth (optional)")
    args = ap.parse_args()

    if args.cmd == "load":
        cmd_load()
    elif args.cmd == "clean":
        cmd_clean()
    elif args.cmd == "model":
        cmd_model(max_depth=args.max_depth)
    elif args.cmd == "evaluate":
        cmd_evaluate()
