"""Tapered rod contact geometry and reciprocal normal response."""
import numpy as np
import pytest
import torch

from genesis.engine.solvers.vbd_rod_contact import closest_tapered, contact_residual

pytestmark = pytest.mark.parametrize('backend', [None], indirect=True)


def test_tapered_segment_contacts():
    a = np.array([[-1., 0., 0.], [1., 0., 0.]])
    b = np.array([[0., -1., .4], [0., 1., .4]])
    wa, wb, gap, lower = closest_tapered(a, np.array([.1,.1]), b, np.array([.1,.1]))
    np.testing.assert_allclose(wa, [.5,.5], atol=1e-9)
    np.testing.assert_allclose(wb, [.5,.5], atol=1e-9)
    assert abs(gap-.2)<1e-12
    # Radius slope moves the witness away from the centreline closest point.
    wa, wb, gap, lower = closest_tapered(a, np.array([.1,.3]), b, np.array([.1,.1]))
    assert wa[1]>.5
    assert gap<.1
    endpoint = np.array([[2.,0.,.4],[3.,0.,.4]])
    wa, wb, gap, lower = closest_tapered(a, np.array([.1,.1]), endpoint, np.array([.1,.1]))
    np.testing.assert_allclose(wa,[0,1],atol=1e-9)
    np.testing.assert_allclose(wb,[1,0],atol=1e-9)
    triangle=np.array([[-2.,-2.,.4],[2.,-2.,.4],[0.,2.,.4]])
    wa,wb,gap,lower=closest_tapered(a,np.array([.1,.1]),triangle,np.zeros(3))
    assert abs(gap-.3)<1e-10
    assert lower <= .3
    assert gap-lower < 1e-6
    assert (wb>=0).all() and abs(wb.sum()-1)<1e-12
    for scale in (1e-3,1e3):
        scaled=closest_tapered(a*scale,np.array([.1,.1])*scale,triangle*scale,np.zeros(3))
        assert abs(scaled.gap/scale-.3)<1e-10
        assert scaled.lower_bound/scale<=.3+1e-14


def test_barrier_contact_response():
    g=torch.tensor([.001,.004],dtype=torch.float64,requires_grad=True)
    k=100.
    residual=contact_residual(g,k,.003)
    energy=residual.square().sum()/2
    grad=torch.autograd.grad(energy,g)[0]
    assert grad[0]<0 and grad[1]==0
    assert torch.autograd.gradcheck(lambda g:contact_residual(g,k,.003),(g,))
    with pytest.raises(ValueError,match='intersect'):
        contact_residual(torch.tensor([-.001]),k,.003)
