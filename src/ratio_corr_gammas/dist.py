import numdifftools as nd
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, hyp2f1
from scipy.stats import rv_continuous


class ratio_of_correlated_gammas(rv_continuous):
    """
    Ratio of Correlated Gammas (RCG) distribution.

    The RCG distribution is defined for the ratio of two correlated gamma-distributed variables.
    beta val ~ RCG(alpha, lambda_m, lambda_u, rho)
    
    Parameters
    ----------
    alpha : float
        Shape parameter, must be greater than 1.
    lambda_m : float
        Rate parameter for the first gamma distribution.
    lambda_u : float
        Rate parameter for the second gamma distribution.
    rho : float, optional
        Correlation coefficient between the two gamma variables, defaults to 0.5.
    """

    def __init__(self, alpha, lambda_m, lambda_u, rho=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha <= 1:
            raise ValueError('alpha must be greater than 1')
        if not (0 <= rho <= 1):
            raise ValueError('rho must be between 0 and 1')

        self.alpha = alpha
        self.lambda_m = lambda_m
        self.lambda_u = lambda_u
        self.theta = lambda_m / lambda_u  # ratio of rates, lam_M/lam_U = E_U/E_M
        self.rho = rho

        self._check_pdf()


    @property
    def expect_theta(self):
        """
        Expected value of theta (E[U/M]).

        See Berger M, Wagner M & Schmid M, 2019 for proof.

        Returns
        -------
        float
            The expected value of theta.
        """
        return self.theta * self._C

    @property
    def _C(self):
        """
        Internal constant used in the computation of expect_theta.

        Returns
        -------
        float
            The computed constant value.
        """
        return gamma(self.alpha + 1) * gamma(self.alpha - 1) / gamma(self.alpha) ** 2 * hyp2f1(1, 1, self.alpha, self.rho)

    @property
    def expect_marginals(self):
        """
        Expected values of the marginal gamma distributions.

        Returns
        -------
        dict
            Expected values for X_m and X_u, X_* ~ Gamma(alpha, lambda_*).
        """
        return {'X_m': self.alpha / self.lambda_m, 'X_u': self.alpha / self.lambda_u}

    @property
    def expect_b_marginal(self):
        """
        Expected value of the b marginal distribution.

        Returns
        -------
        float
            The expected value of the b marginal distribution.
        """
        return self.expect_marginals['X_m'] / sum(self.expect_marginals.values())

    def expect(self, func=None, args=(), loc=0, scale=1, lb=0, ub=1, conditional=False, **kwds):
        """
        Overwrites the expect method from rv_continuous.

        Parameters
        ----------
        func : callable, optional
            Function for which the expectation is computed.
        lb, ub : float, optional
            Lower and upper bounds for integration, respectively.
        Other parameters are inherited from rv_continuous.

        Returns
        -------
        float
            The computed expectation value.
        """

        if func is None:
            def fun(x, *args):
                return x * self.pdf(x, *args)
        else:
            def fun(x, *args):
                return func(x) * self.pdf(x, *args)

        vals = quad(fun, lb, ub, **kwds)[0]
        return np.array(vals)[()]

    def pdf(self, x):
        """
        Probability density function of the RCG distribution at x.

        Parameters
        ----------
        x : float
            Value at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function evaluated at x.
        """
        return self._pdf(x)

    def _pdf(self, betaval):
        """
        Internal method for computing the pdf.
        
        See Weinhold L, et al. 2016 for proof.

        Parameters
        ----------
        betaval : float
            Value at which to evaluate the pdf.

        Returns
        -------
        float
            The probability density function evaluated at betaval.
        """
        one_minus_betaval = 1 - betaval

        dens = 0
        dens += gamma(2 * self.alpha) / gamma(self.alpha) ** 2
        dens *= (self.lambda_m * self.lambda_u) ** self.alpha
        dens *= (1 - self.rho) ** self.alpha
        dens *= np.power(betaval * one_minus_betaval, self.alpha - 1)
        dens *= self.lambda_m * betaval + self.lambda_u * one_minus_betaval
        dens /= np.power(
            np.power(self.lambda_m * betaval + self.lambda_u * one_minus_betaval, 2) 
            - 4 * self.rho * self.lambda_m * self.lambda_u * betaval * one_minus_betaval,
            self.alpha + 0.5
        )
        
        return dens

    def dpdf(self, x):
        """
        Derivative of the probability density function at x.

        Parameters
        ----------
        x : float
            Value at which to evaluate the derivative of the pdf.

        Returns
        -------
        float
            The derivative of the probability density function evaluated at x.
        """
        # Handles (b(1-b))^(alpha-1) when b=0 or b=1
        if x in [0, 1]:
            return 0

        return nd.Derivative(self._pdf)(x)

    def _check_pdf(self):
        """
        Verifies that the integral of the pdf over its domain is 1.

        Raises
        ------
        AssertionError
            If the integral of the pdf is not approximately 1.
        """
        I, _ = quad(self.pdf, 0, 1)
        np.testing.assert_almost_equal(I, 1, decimal=5)