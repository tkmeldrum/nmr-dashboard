"""matplotlib figures (rendered to PNG so the page can show them and, later, save them)."""
import base64
import io

import numpy as np


def corner_png(samples, labels, truths=None, fg="#1a1a1a", grid="#cccccc", accent="#0072B2", mark="#D55E00", bins=40):
    """Corner plot of posterior samples (n_samples, k): marginal histograms on the diagonal, 2-D densities below.

    truths : optional length-k values drawn as lines (e.g. the least-squares fit). Returns base64 PNG with a
    transparent background; fg/grid/accent are CSS colours so the figure matches the page theme.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    samples = np.asarray(samples, float)
    k = samples.shape[1]
    size = max(5.0, 1.7 * k + 1)
    fig, axes = plt.subplots(k, k, figsize=(size, size), squeeze=False)
    fig.patch.set_alpha(0)
    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            ax.patch.set_alpha(0)
            for side in ax.spines.values():
                side.set_color(grid)
            ax.tick_params(colors=fg, labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, color=accent, alpha=0.85, histtype="stepfilled")
                ax.set_yticks([])
                if truths is not None and np.isfinite(truths[i]):
                    ax.axvline(truths[i], color=mark, lw=1.2)
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap="viridis", cmin=1)
                if truths is not None and np.isfinite(truths[j]) and np.isfinite(truths[i]):
                    ax.plot(truths[j], truths[i], "s", color=mark, ms=4)
                    ax.axvline(truths[j], color=mark, lw=0.8, alpha=0.7)
                    ax.axhline(truths[i], color=mark, lw=0.8, alpha=0.7)
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[j], color=fg, fontsize=9)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(30)
            if j > 0:
                ax.set_yticklabels([])
            elif i > 0:
                ax.set_ylabel(labels[i], color=fg, fontsize=9)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
