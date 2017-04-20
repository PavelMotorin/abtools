from scipy.stats.kde import gaussian_kde

def KL(a, b):

    x = np.linspace(
        min(a.min(), b.min()) - 1,
        max(a.max(), b.max()) + 1,
        100
    )

    p = gaussian_kde(a)(x)
    q = gaussian_kde(b)(x)

    p = p/np.sum(p)
    q = q/np.sum(q)

    return np.sum(np.where(p != 0, (p) * np.log(p / q), 0))
