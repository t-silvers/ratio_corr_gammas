import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import uniform


class RejectionSamplerRCG:
    """
    Rejection sampling for Ratio of Correlated Gammas (RCG) distribution.

    Parameters
    ----------
    dist : rv_continuous
        An instance of a scipy.stats rv_continuous object representing the RCG distribution.
    domain : list, optional
        The domain over which the distribution is defined, default is [0, 1].

    Attributes
    ----------
    M : float
        Scale parameter for the rejection sampler.
    efficiency : float
        Efficiency of the sampler, calculated as 1 / M.

    Methods
    -------
    rvs(size, random_state=None)
        Generates random variates using rejection sampling.
    """
    def __init__(self, dist, domain=[0, 1]):
        self.dist = dist
        self.domain = domain
        self.M = self._calculate_M()
        self.efficiency = 1 / self.M

        if self.efficiency < 0.01:
            raise ValueError('Estimated efficiency is too low. Consider reducing the scale parameter by setting a custom value for `M`.')

    @property
    def M(self):
        """Scale parameter for rejection sampler."""
        return self._M

    @M.setter
    def M(self, value):
        self._M = value

    def rvs(self, size, random_state=None):
        """
        Generates random variates using rejection sampling.
        
        Note that this does not guarantee `size` samples.

        Parameters
        ----------
        size : int
            The number of random variates to generate.
        random_state : int or np.random.Generator, optional
            A seed or random state for reproducible output.

        Returns
        -------
        np.ndarray
            An array of random variates.

        Raises
        ------
        ValueError
            If the generated number of samples is insufficient.
        """
        urng = np.random.default_rng(random_state)

        oversample_scale = 10
        
        # Calculate efficiency-adjusted size
        size_ = int(size * self.M) * oversample_scale
        if size_ >= np.finfo(np.float32).max:
            raise ValueError('Size is too large')

        u1 = uniform.rvs(size=size_, random_state=urng)
        f_u1 = self.dist.pdf(u1)
        
        u2 = uniform.rvs(size=size_, random_state=urng)
        idx = u2 <= f_u1 / self.M
        
        v = u1[idx]

        if len(v) < size:
            # TODO: Better handling
            print('Warning: Not enough samples. Consider increasing `size` (and post-sampling) or adjusting the scale parameter `M`.')
        
        return v[:size]

    def _calculate_M(self):
        """
        Calculate the scale parameter M for the rejection sampler.
        
        Uses the Brent method to find a local minimum in the interval 0 < xopt < 1. 

        Returns
        -------
        float
            The calculated scale parameter.
        """
        res = minimize_scalar(lambda x: -self.dist.pdf(x), bounds=(0, 1), method='bounded')
        return -res.fun
    
    def __repr__(self):
        return 'RejectionSamplerRCG()'