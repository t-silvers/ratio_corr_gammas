from scipy.stats.sampling import SimpleRatioUniforms, TransformedDensityRejection

from ratio_corr_gammas.dist import ratio_of_correlated_gammas
from ratio_corr_gammas.rejection_sampler import RejectionSamplerRCG


def simulate_betavals_rcg(size, dist=None, theta=None, alpha=2., rho=0.45, scale=1., random_state=None, sampler=None):
    """
    Convenience function to simulate DNA methylation beta values from the RCG distribution.

    Parameters
    ----------
    size : int
        The number of beta values to simulate.
    dist : ratio_of_correlated_gammas, optional
        An instance of the RCG distribution. If None, it will be created based on `theta`, `alpha`, `rho`, and `scale`.
    theta : float, optional
        Ratio of expected un-methylated to methylated probe intensity, i.e. E_U / E_M, where
           E_U / E_M = (alpha / lambda_u) / (alpha / lambda_m) = lambda_m / lambda_u.
    alpha : float, default=2.0
        Shape parameter of the RCG distribution.
    rho : float, default=0.45
        Correlation coefficient between the two gamma random variables. Default is 0.45, which is average correlation found in Weinhold L, 2016.
    scale : float, default=1.0
        Scale parameter for the RCG distribution.
    random_state : int or np.random.Generator, optional
        A seed or random state for reproducible output.
    sampler : {'sru', 'tdr', 'rej'}, optional
        Type of sampler to use. If None, an appropriate sampler is selected based on `alpha`.

    Returns
    -------
    np.ndarray
        An array of simulated beta values.

    Raises
    ------
    ValueError
        If neither `dist` nor `theta` is specified, or if `theta` is not positive.
    TypeError
        If `dist` is not an instance of `ratio_of_correlated_gammas`.
    """

    samplers = {
        'sru': SimpleRatioUniforms,
        'tdr': TransformedDensityRejection,
        'rej': RejectionSamplerRCG,
    }

    if dist is None:
        if theta is None:
            raise ValueError('Either `dist` or `theta` must be specified')
        if theta <= 0:
            raise ValueError('theta must be positive')
        dist = ratio_of_correlated_gammas(alpha, scale * max(theta, 1.), scale * 1. / min(theta, 1.), rho, name='RCG')
    elif not isinstance(dist, ratio_of_correlated_gammas):
        raise TypeError("`dist` must be an instance of `ratio_of_correlated_gammas`")

    if sampler:
        sampler = samplers[sampler](dist, domain=[0, 1])
    else:
        if alpha > 1:
            sampler = TransformedDensityRejection(dist, domain=[0, 1])
        else:
            sampler = RejectionSamplerRCG(dist)
            if sampler.efficiency < 0.01:
                sampler = SimpleRatioUniforms(dist, domain=[0, 1])

    try:
        samples = sampler.rvs(size=size, random_state=random_state)
    except Exception as e:
        print(f"Error encountered during sampling: {e}. Attempting alternative sampling method.")
        sampler = RejectionSamplerRCG(dist)
        sampler.M = 1_000
        samples = sampler.rvs(size=size, random_state=random_state)

    # TODO: Return named tuple with samples and sampler
    # print(sampler)

    return samples