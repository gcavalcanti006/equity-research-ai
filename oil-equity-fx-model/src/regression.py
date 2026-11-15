"""OLS regression: equity return = a + b1*oil return + b2*fx return + error."""
import statsmodels.api as sm


def run_ols(returns, extra_cols=()):
    """Fit OLS. `extra_cols` are additional regressors (e.g. a lagged oil return)."""
    x_cols = ["oil", "fx", *extra_cols]
    X = sm.add_constant(returns[x_cols])
    y = returns["equity"]
    # HAC (Newey-West) standard errors are robust to autocorrelation/heteroskedasticity.
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def ols_table(result):
    """Coefficients, standard errors, t-stats and p-values as a small table."""
    import pandas as pd
    return pd.DataFrame({"coef": result.params, "std_err": result.bse,
                         "t_stat": result.tvalues, "p_value": result.pvalues})
