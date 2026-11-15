"""Machine-learning comparison with a strictly time-ordered train/test split."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def time_split(df, test_fraction):
    """First (1 - test_fraction) of the days = train, the rest = test. No shuffling."""
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def make_model(model_type, seed):
    if model_type == "random_forest":
        # min_samples_leaf keeps trees from memorising noise
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                     random_state=seed)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                         learning_rate=0.05, random_state=seed)
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(returns, x_cols, model_type, test_fraction, seed):
    """Fit OLS and the ML model on train only; score all on the test period.

    Also scores a naive baseline that always predicts the TRAIN mean.
    Out-of-sample R2 can be negative: that means worse than predicting the mean.
    """
    train, test = time_split(returns, test_fraction)
    Xtr, ytr = train[x_cols], train["equity"]
    Xte, yte = test[x_cols], test["equity"]

    ml = make_model(model_type, seed).fit(Xtr, ytr)
    ols = LinearRegression().fit(Xtr, ytr)

    preds = {
        "ml": ml.predict(Xte),
        "ols": ols.predict(Xte),
        "baseline_train_mean": np.full(len(yte), ytr.mean()),
    }
    metrics = {name: {"r2_test": r2_score(yte, p),
                      "rmse_test": float(np.sqrt(mean_squared_error(yte, p)))}
               for name, p in preds.items()}
    importances = dict(zip(x_cols, getattr(ml, "feature_importances_", [])))
    return {"train": train, "test": test, "preds": preds,
            "metrics": metrics, "importances": importances}
