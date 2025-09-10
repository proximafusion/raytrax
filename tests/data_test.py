from raytrax.data import get_w7x_wout

def test_w7x_wout():
    """Test that the W7-X equilibrium fixture works correctly."""
    w7x_wout = get_w7x_wout()
    # Check that the equilibrium has the expected properties
    assert w7x_wout.ns > 0
    assert w7x_wout.nfp == 5  # W7-X has 5-fold symmetry
    assert hasattr(w7x_wout, 'lasym')
    
    # Check that arrays have the right shapes
    assert len(w7x_wout.xm.shape) == 1  # 1D array
    assert len(w7x_wout.xn.shape) == 1  # 1D array
    assert len(w7x_wout.xm_nyq.shape) == 1  # 1D array
    assert len(w7x_wout.xn_nyq.shape) == 1  # 1D array
    
    # Check that 2D arrays have the right shapes
    assert len(w7x_wout.rmnc.shape) == 2
    assert w7x_wout.rmnc.shape[1] == w7x_wout.ns
    assert len(w7x_wout.zmns.shape) == 2
    assert w7x_wout.zmns.shape[1] == w7x_wout.ns
    assert len(w7x_wout.bsupumnc.shape) == 2
    assert w7x_wout.bsupumnc.shape[1] == w7x_wout.ns
    assert len(w7x_wout.bsupvmnc.shape) == 2
    assert w7x_wout.bsupvmnc.shape[1] == w7x_wout.ns
    
    # Check that we have some additional fields that would be in VmecWOut but not TestWout
    assert hasattr(w7x_wout, 'ier_flag')
    assert hasattr(w7x_wout, 'mpol')
    assert hasattr(w7x_wout, 'ntor')
