"""Multi-factor model, honest evaluation (walk-forward) and tests against simple baselines.

Walk-forward = at each step the model is fitted on the PAST only and predicts the next block of
days, then the window grows. No model ever sees the data it is scored on.

The benchmarks every model must beat to be worth anything:
  - 'mean'  : always predict the average return seen so far
  - 'capm'  : the stock explained by its market index only (Ibovespa for Brazil, S&P 500 for US)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

Z80 = 1.2816     # a +-1.2816 sigma band holds 80% of a normal distribution


class FactorModel:
    """Linear model. Optionally 'cleans' each non-market factor of its market part first.

    Why: Ibovespa, USD/BRL and yields move together. Without cleaning, the Brent coefficient
    would partly be a market coefficient. The cleaning is learned on training data only."""

    def __init__(self, market_cols, other_cols, orthogonalize=True, method="ols", alpha=10.0):
        self.market_cols, self.other_cols = list(market_cols), list(other_cols)
        self.orthogonalize, self.method, self.alpha = orthogonalize, method, alpha

    def _transform(self, X):
        Z = X[self.market_cols + self.other_cols].copy()
        if self.orthogonalize and self.market_cols:
            M = np.column_stack([np.ones(len(X)), X[self.market_cols].values])
            for c in self.other_cols:
                Z[c] = X[c].values - M @ self.ortho_[c]
        return Z

    def fit(self, X, y):
        self.ortho_ = {}
        if self.orthogonalize and self.market_cols:
            M = np.column_stack([np.ones(len(X)), X[self.market_cols].values])
            for c in self.other_cols:
                self.ortho_[c] = np.linalg.lstsq(M, X[c].values, rcond=None)[0]
        Z = self._transform(X)
        if self.method == "ridge":
            self.scaler_ = StandardScaler().fit(Z.values)
            self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(Z.values), y.values)
        else:
            self.model_ = LinearRegression().fit(Z.values, y.values)
        return self

    def predict(self, X):
        Z = self._transform(X).values
        if self.method == "ridge":
            Z = self.scaler_.transform(Z)
        return self.model_.predict(Z)

    def coefficients(self):
        names = self.market_cols + self.other_cols
        return dict(zip(names, np.ravel(self.model_.coef_)))


class MeanModel:
    """Predicts the training average, always."""
    def fit(self, X, y):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def default_models(market_cols, other_cols, cfg, sector_cols=(), fx_cols=()):
    """All models that go through the same walk-forward test.

    mean           : the running average (no information)
    capm           : market index only
    sector_fx      : like the ORIGINAL project: sector commodity + exchange rate, NO market index
    factors        : market + every factor (the full model)
    factors_nolag  : the full model without the 'yesterday' (_lag1) columns
    factors_ridge  : the full model with shrinkage (ridge)"""
    no_lag = [c for c in other_cols if not c.endswith("_lag1")]
    old_spec = list(sector_cols) + list(fx_cols)
    models = {
        "mean": lambda: MeanModel(),
        "capm": lambda: FactorModel(market_cols, [], orthogonalize=False),
        "factors": lambda: FactorModel(market_cols, other_cols, orthogonalize=cfg["orthogonalize"]),
        "factors_nolag": lambda: FactorModel(market_cols, no_lag, orthogonalize=cfg["orthogonalize"]),
        "factors_ridge": lambda: FactorModel(market_cols, other_cols, orthogonalize=cfg["orthogonalize"],
                                             method="ridge", alpha=cfg["ridge_alpha"]),
    }
    if old_spec:
        models["sector_fx"] = lambda: FactorModel([], old_spec, orthogonalize=False)
    return models


def walk_forward(X, y, factories, min_train, step):
    """Expanding-window predictions. Returns a long table: date, model, pred, sigma, actual.

    sigma = standard deviation of the model's errors on its TRAINING data (used for intervals)."""
    rows = []
    n = len(y)
    for start in range(min_train, n, step):
        end = min(start + step, n)
        Xtr, ytr, Xte, yte = X.iloc[:start], y.iloc[:start], X.iloc[start:end], y.iloc[start:end]
        for name, make in factories.items():
            model = make().fit(Xtr, ytr)
            sigma = float(np.std(ytr.values - model.predict(Xtr), ddof=1))
            pred = model.predict(Xte)
            rows.append(pd.DataFrame({"date": yte.index, "model": name, "pred": pred,
                                      "sigma": sigma, "actual": yte.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date", "model", "pred", "sigma", "actual"])


def dm_test(err_a, err_b, lags=5):
    """Diebold-Mariano: is model A's squared error different from B's?
    Returns (mean loss difference B-A, p-value). Positive mean = A is better."""
    d = np.asarray(err_b) ** 2 - np.asarray(err_a) ** 2
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.pvalues[0])


def evaluate(preds, hac_lags=5):
    """One row per model: out-of-sample R2 vs the running mean, RMSE, hit rate, 80% interval coverage,
    and Diebold-Mariano p-values (factors vs capm, capm vs mean)."""
    piv = {m: g.set_index("date") for m, g in preds.groupby("model")}
    actual = piv["mean"]["actual"]
    sse_mean = float(((actual - piv["mean"]["pred"]) ** 2).sum())
    rows = []
    for name, g in piv.items():
        err = g["actual"] - g["pred"]
        nonzero = (g["pred"] != 0) & (g["actual"] != 0)
        hit = float((np.sign(g["pred"][nonzero]) == np.sign(g["actual"][nonzero])).mean())
        rows.append({"model": name, "n_test": len(g),
                     "r2_oos_vs_mean": 1 - float((err ** 2).sum()) / sse_mean,
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "hit_rate": hit,
                     "coverage_80": float((err.abs() <= Z80 * g["sigma"]).mean())})
    table = pd.DataFrame(rows).set_index("model")
    table["dm_p_vs_capm"] = np.nan
    table["dm_gain_vs_capm"] = np.nan
    if "capm" in piv:
        err_capm = piv["capm"]["actual"] - piv["capm"]["pred"]
        for name in table.index:
            if name in ("capm", "mean"):
                continue
            gain, p = dm_test(piv[name]["actual"] - piv[name]["pred"], err_capm, hac_lags)
            table.loc[name, ["dm_gain_vs_capm", "dm_p_vs_capm"]] = [gain, p]
    # Is the full model better than each simpler model? (positive gain and small p = yes)
    table["dm_p_factors_over_it"] = np.nan
    table["dm_gain_factors_over_it"] = np.nan
    if "factors" in piv:
        err_full = piv["factors"]["actual"] - piv["factors"]["pred"]
        for name in table.index:
            if name in ("factors", "mean"):
                continue
            gain, p = dm_test(err_full, piv[name]["actual"] - piv[name]["pred"], hac_lags)
            table.loc[name, ["dm_gain_factors_over_it", "dm_p_factors_over_it"]] = [gain, p]
    return table


def full_sample_report(X, y, market_cols, other_cols, orthogonalize, hac_lags=5):
    """In-sample OLS with HAC standard errors, to read the coefficients (not to judge accuracy)."""
    fm = FactorModel(market_cols, other_cols, orthogonalize=orthogonalize).fit(X, y)
    Z = fm._transform(X)
    res = sm.OLS(y, sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    table = pd.DataFrame({"coef": res.params, "std_err": res.bse, "t_stat": res.tvalues,
                          "p_value": res.pvalues})
    return table, float(res.rsquared), float(res.rsquared_adj), fm


def vif_table(X):
    """Variance inflation factor: above ~5 means a factor is mostly explained by the others."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    A = sm.add_constant(X).values
    return pd.Series([variance_inflation_factor(A, i + 1) for i in range(X.shape[1])],
                     index=X.columns, name="vif")


def rolling_coefficients(X, y, fitted_model, window, every=5):
    """How the coefficients drift over time (OLS on a moving window, every `every` days)."""
    Z = fitted_model._transform(X)
    cols = list(Z.columns)
    out = {}
    for end in range(window, len(y) + 1, every):
        A = np.column_stack([np.ones(window), Z.values[end - window:end]])
        beta = np.linalg.lstsq(A, y.values[end - window:end], rcond=None)[0]
        out[y.index[end - 1]] = dict(zip(["const"] + cols, beta))
    return pd.DataFrame(out).T
