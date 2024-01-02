import numpy as np
import pytest
from scipy.stats import kstest
from ratio_corr_gammas.dist import ratio_of_correlated_gammas
from ratio_corr_gammas.rejection_sampler import RejectionSamplerRCG
from ratio_corr_gammas.sample import simulate_betavals_rcg


def test_ratio_of_correlated_gammas_creation():
    """Test creation and basic properties of ratio_of_correlated_gammas distribution."""
    alpha, lambda_m, lambda_u, rho = 2.0, 1.0, 0.5, 0.45
    dist = ratio_of_correlated_gammas(alpha, lambda_m, lambda_u, rho)
    assert dist.alpha == alpha
    assert dist.lambda_m == lambda_m
    assert dist.lambda_u == lambda_u
    assert dist.rho == rho


def test_ratio_of_correlated_gammas_pdf():
    """Test the PDF method of the ratio_of_correlated_gammas distribution."""
    dist = ratio_of_correlated_gammas(2.0, 1.0, 0.5, 0.45)
    x = 0.5
    assert isinstance(dist.pdf(x), float)


def test_rejection_sampler_rcg():
    """Test initialization and sampling of the RejectionSamplerRCG."""
    dist = ratio_of_correlated_gammas(2.0, 1.0, 0.5, 0.45)
    sampler = RejectionSamplerRCG(dist)
    samples = sampler.rvs(size=100)
    assert len(samples) == 100
    assert all(0 <= val <= 1 for val in samples)


def test_simulate_betavals_rcg():
    """Test the simulate_betavals_rcg function."""
    size = 100
    samples = simulate_betavals_rcg(size, alpha=2.0, rho=0.45, scale=1.0)
    assert len(samples) == size
    # GoF test (e.g., Kolmogorov-Smirnov) to check if samples follow the expected distribution
    dist = ratio_of_correlated_gammas(2.0, 1.0, 0.5, 0.45)
    stat, p_value = kstest(samples, dist.cdf)
    assert p_value > 0.05