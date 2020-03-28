
# quick routine to validate my calculus for the better-than-noise threshold
def monte_carlo(iter_num):
    """

    :param iter_num: Number of iterations for the Monte Carlo simulation.
    :return: mse
    """
    v = np.random.random((iter_num, 2))
    d1 = v[:, 0] - v[:, 1]  # random guess
    d2 = v[:, 0] - 0.5  # fixed guess of 0.5
    s = d2 ** 2
    mse = np.mean(s)
    return mse