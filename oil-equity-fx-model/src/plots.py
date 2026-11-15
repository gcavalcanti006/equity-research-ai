"""Charts saved as PNG."""
import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt


def plot_rolling_corr(roll, window, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(roll.index, roll["equity_vs_oil"], label="Equity vs Brent")
    ax.plot(roll.index, roll["equity_vs_fx"], label="Equity vs USD/BRL")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Rolling {window}-day correlation of daily log returns")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(test, preds, model_label, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(test.index, test["equity"], label="Actual", lw=0.8, color="grey")
    ax.plot(test.index, preds["ols"], label="OLS prediction", lw=1.2)
    ax.plot(test.index, preds["ml"], label=f"{model_label} prediction", lw=1.2)
    ax.set_title("Predicted vs actual daily log returns (test period only)")
    ax.set_ylabel("Log return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
