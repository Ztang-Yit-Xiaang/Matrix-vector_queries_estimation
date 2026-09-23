"""Small exhaustive regressions for the report; not Monte Carlo experiments."""
from fractions import Fraction
from itertools import product

import numpy as np
import pytest


def signs(d):
    return np.array(list(product((-1.0, 1.0), repeat=d)))


@pytest.mark.parametrize('d', [2, 3, 4, 5])
def test_rademacher_variance_and_corrected_fourth_moment(d):
    rng = np.random.default_rng(17000 + d)
    c = rng.normal(size=(d, d))
    c = (c + c.T) / 2
    np.fill_diagonal(c, 0)
    g = signs(d)
    z = np.einsum('ni,ij,nj->n', g, c, g)
    sigma2 = 2 * np.sum(c * c)
    assert np.allclose(c, c.T)
    assert np.mean(z) == pytest.approx(0, abs=1e-14)
    assert np.var(z) == pytest.approx(sigma2)
    c2 = c @ c
    mu4 = (3 * sigma2**2 + 48 * np.trace(c2 @ c2)
           - 96 * np.sum(np.diag(c2)**2) + 32 * np.sum(c**4))
    assert np.mean(z**4) == pytest.approx(mu4)
    kappa = np.linalg.norm(c, 2) / np.linalg.norm(c, 'fro')
    assert mu4 <= (3 + 12*kappa*kappa) * sigma2**2 + 1e-10


@pytest.mark.parametrize('s', [2, 3, 4, 5])
def test_exact_sample_variance_moments_and_hoeffding(s):
    support = np.array([-2., 0., 5.])
    mu = support.mean()
    sigma2 = np.mean((support-mu)**2)
    mu4 = np.mean((support-mu)**4)
    x = np.array(list(product(support, repeat=s)))
    z = x-mu
    sv = np.var(x, axis=1, ddof=1)
    assert sv.mean() == pytest.approx(sigma2)
    assert sv.var() == pytest.approx((mu4-(s-3)/(s-1)*sigma2**2)/s)
    linear = (z*z-sigma2).mean(axis=1)
    canonical = sum(-z[:, i]*z[:, j] for i in range(s) for j in range(i+1, s)) / (s*(s-1)/2)
    assert np.allclose(sv-sigma2, linear+canonical)
    allpairs = sum((x[:, i]-x[:, j])**2 for i in range(s) for j in range(i+1, s)) / (s*(s-1))
    assert np.allclose(allpairs, sv)
    assert np.allclose(np.outer(support-mu, support-mu).mean(axis=1), 0)
    assert np.var(((support-mu)**2-sigma2)/2) > 0


def test_zero_variance_and_pair_mean():
    x = np.array(list(product([-1., 2., 7.], repeat=2)))
    w = (x[:, 0]-x[:, 1])**2/2
    assert w.mean() == pytest.approx(np.var([-1., 2., 7.]))
    assert np.var(np.full(8, 9.), ddof=1) == 0


def test_structured_energy_and_graph_principal_angle():
    rng = np.random.default_rng(17100)
    d, k, eta = 9, 3, .03
    u = np.linalg.qr(rng.normal(size=(d, d)))[0]
    s = rng.normal(size=(d, k))
    a = eta*np.eye(d)+(1-eta)*u[:, :k]@u[:, :k].T
    q = np.linalg.qr(a@s)[0]
    r = np.eye(d)-q@q.T
    z = r@u[:, :k]
    h = r@a@r
    energy = eta**2*(d-k)+2*eta*(1-eta)*np.sum(z*z)+(1-eta)**2*np.sum((z.T@z)**2)
    assert energy == pytest.approx(np.sum(h*h))
    f = eta*(u[:, k:].T@s)@np.linalg.inv(u[:, :k].T@s)
    f2 = np.linalg.norm(f, 2)
    assert np.linalg.norm(z, 2) == pytest.approx(f2/np.sqrt(1+f2*f2))


def test_discrete_missing_direction_transcript():
    eta = .01
    a = np.diag([1., 1., eta, eta])
    s = np.array([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
    w = np.array([1., 1., 0., 0.])/np.sqrt(2)
    q = np.linalg.qr(a@s)[0]
    altered = a-(1-eta)*np.outer(w, w)
    assert np.allclose(q.T@w, 0)
    assert np.allclose(a@s, altered@s)
    assert np.allclose(a@q, altered@q)
    r = np.eye(4)-q@q.T
    assert np.allclose(r@a@r@w, w)


def test_rademacher_energy_can_increase_under_nesting():
    q = np.ones((2, 1))/np.sqrt(2)
    h = np.eye(2)-q@q.T
    assert np.sum((h-np.diag(np.diag(h)))**2) == pytest.approx(.5)
    assert np.sum((np.eye(2)-np.diag(np.ones(2)))**2) == 0


@pytest.mark.parametrize('energy,next_energy,D', [(3., 2., 10), (1., .99, 130), (0., 0., 9), (0., 1., 9)])
def test_success_and_failure_marginals(energy, next_energy, D):
    direct = 2*next_energy/(D-2)-2*energy/D
    formula = -2*(D*(energy-next_energy)-2*energy)/(D*(D-2))
    assert direct == pytest.approx(formula)
    if energy > 0:
        assert (direct < 0) == (next_energy/energy < (D-2)/D)
    assert 2*energy/(D-1)-2*energy/D == pytest.approx(2*energy/(D*(D-1)))


def test_paired_risk_difference_unbiased_by_enumeration():
    g = signs(3)
    a = np.array([[1., .2, .3], [.2, 2., .1], [.3, .1, 1.]])
    b = np.array([[1., -.3, .6], [-.3, 1., .5], [.6, .5, 2.]])
    xa = np.einsum('ni,ij,nj->n', g, a, g)
    xb = np.einsum('ni,ij,nj->n', g, b, g)
    paired = [(xa[i]-xa[j])**2/(2*12)-(xb[i]-xb[j])**2/(2*16)
              for i, j in product(range(len(g)), repeat=2)]
    assert np.mean(paired) == pytest.approx(np.var(xa)/12-np.var(xb)/16)


def test_chebyshev_vacuity_and_exact_41_42_boundary():
    eps = [np.sqrt(2*(80+2/(s-1))/(s*.05)) for s in range(2, 33)]
    assert all(x > y for x, y in zip(eps, eps[1:]))
    assert eps[-1] > 10
    delta = Fraction(1, 20)
    assert delta*41*40 == 2*40+2
    assert delta*42*41 > 2*41+2
    assert all(delta*n*(n-1) <= 2*(n-1)+2 for n in range(2, 42))


def test_truncation_bound_floor_is_not_a_probability_claim():
    for cap, beta, eps, s in product([64, 128, 1024], [0., .01, .2], [.25, .75, .99], [4, 8, 16, 32]):
        nu = min(81., cap**2/4, 81-max(0, 1-beta)**2)
        assert 80 <= nu <= 81
        if beta < eps:
            x = eps-beta
            bound = np.exp(-s*x*x/(2*(nu+x/3)))
            assert bound > np.exp(-s/160) >= np.exp(-.2)
            assert bound > .05


def test_paid_fallback_and_directed_safety_are_distinct():
    original, paid = 100, 80
    assert (1/paid)/(1/original) > 1
    eps_candidate, eps_baseline = .2, 3.
    threshold = (1-eps_candidate)/(1+eps_baseline)*paid/original
    assert threshold == pytest.approx(.16)
    # Baseline upper-deviation radius >1 remains algebraically admissible.
    assert threshold/(1-eps_candidate)/paid == pytest.approx(1/(1+eps_baseline)/original)
