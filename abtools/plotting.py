import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def ppplot(a, b, distr_names=['First', 'Second'], num_points=1000):
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

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(a_p, b_p, s=30, c=sns.xkcd_rgb["denim blue"])
    plt.plot([0, 1], [0, 1], c=sns.xkcd_rgb["pale red"], alpha=0.75)
    plt.title('P-P plot')
    plt.xlabel("%s cumulative distribution" % distr_names[0])
    plt.ylabel("%s cumulative distribution" % distr_names[1])
    fig.show()


def qqplot(a, b, distr_names=['First', 'Second'],
           num_points=1000, fitted_line=False):
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

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(q_a, q_b, s=30, c=sns.xkcd_rgb["denim blue"])
    plt.plot(q_a, line, c=sns.xkcd_rgb["pale red"], alpha=0.75)
    plt.title('Q-Q plot')
    plt.xlabel("%s distribution's percentiles" % distr_names[0])
    plt.ylabel("%s distribution's percentiles" % distr_names[1])
    fig.show()
