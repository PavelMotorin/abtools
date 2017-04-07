import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def ppplot(a, b, distr_names=['First', 'Second'],
           num_points=5000, splitted=False, figsize=(8, 8)):
    """
    P-P plot - plots two cumulative distribution functions against each other.
    Useful for comparing values of two distributions:
    two empirical distributions or theoretical and empirical

    """
    a = np.array(a)
    b = np.array(b)

    a_p, b_p = [], []
    lower_bound = min(a.min(), b.min())
    upper_bound = max(a.max(), b.max())
    z_space = np.linspace(lower_bound, upper_bound, num_points)

    for z in z_space:
        a_p.append((a <= z).mean())
        b_p.append((b <= z).mean())

    def plot(a_p, b_p, ax, q):
        ax.scatter(a_p, b_p, s=30, c=sns.color_palette()[0])
        ax.plot(q, q, c=sns.color_palette()[2], alpha=0.75)
        return ax

    if not splitted:
        fig, ax = plt.subplots(figsize=figsize)
        ax = plot(a_p, b_p, ax, q=[0, 1])
        ax.set_title('P-P plot')
        ax.set_xlabel("%s cumulative distribution" % distr_names[0])
        ax.set_ylabel("%s cumulative distribution" % distr_names[1])
        return ax
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        quartiles = [[.0, .25], [.25, .50], [.50, .75], [.75, 1]]

        for q, ax in zip(quartiles, np.ravel(axs)):
            ax = plot(
                list(filter(lambda z: 1 if z >= q[0] and z < q[1] else 0, a_p)),
                list(filter(lambda z: 1 if z >= q[0] and z < q[1] else 0, b_p)),
                ax,
                q
            )
        plt.suptitle('P-P plots for each quartiles')
        return fig, axs


def qqplot(a, b, distr_names=['First', 'Second'],
           num_points=5000, fitted_line=False, splitted=False, figsize=(8, 8)):
    """
    Q-Q plot - plots quantiles of two distributions against each other.
    Useful for comparing shape of distributions:
    two empirical distributions or theoretical and empirical


    """
    a = np.array(a)
    b = np.array(b)

    q_a = np.array([np.percentile(a, q/(num_points/100))
                    for q in range(0, num_points + 1)])
    q_b = np.array([np.percentile(b, q/(num_points/100))
                    for q in range(0, num_points + 1)])

    if fitted_line:
        coeff = np.polyfit(q_a, q_b, 1)
        line = np.array(q_a) * coeff[0] + coeff[1]
    else:
        line = q_a

    def plot(q_a, q_b, line, ax):
        ax.scatter(q_a, q_b, s=30, c=sns.xkcd_rgb["denim blue"])
        ax.plot(q_a, line, c=sns.xkcd_rgb["pale red"], alpha=0.75)
        return ax

    if not splitted:
        fig, ax = plt.subplots(figsize=figsize)
        ax = plot(q_a, q_b, line, ax)
        ax.set_title('Q-Q plot')
        ax.set_xlabel("%s distribution's quantiles" % distr_names[0])
        ax.set_ylabel("%s distribution's quantiles" % distr_names[1])
        return ax
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        quartiles = [[.0, .25], [.25, .50], [.50, .75], [.75, 1]]

        for q, ax in zip(quartiles, np.ravel(axs)):
            start = int(q[0] * len(q_a))
            stop = int(q[1] * len(q_a))
            ax = plot(
                q_a[start:stop],
                q_b[start:stop],
                line[start:stop],
                ax
            )
        plt.suptitle('Q-Q plots for each quartiles')
        return fig, axs
