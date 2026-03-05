"""
Working pytest for GitlImage - mocks everything, tests the interface
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import corgihowfsc

from howfsc.model.mode import CoronagraphMode

KNOWN_CFGS = [
    ("nfov_band1", "360deg"),
    ("nfov_band1", "half_top"),
    ("spec_band2", "both_sides"),
    ("spec_band3", "both_sides"),
    ("wfov_band4", "360deg"),
]

@pytest.fixture(params=KNOWN_CFGS)
def cfg(request):
    """Load a real CoronagraphMode for each known (mode, dark_hole)."""
    mode, dark_hole = request.param

    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

    modelpath = os.path.join(
        howfscpath,
        "model",
        mode,
        f"{mode}_{dark_hole}",
    )

    cfgfile = os.path.join(modelpath, "howfsc_optical_model.yaml")

    assert os.path.exists(cfgfile), f"Missing cfg file: {cfgfile}"

    cfg = CoronagraphMode(cfgfile)

    return cfg
    
# Mock ALL the external dependencies before any imports
def setup_mocks():
    mocks = [
        'howfsc', 'howfsc.model', 'howfsc.model.mode', 
        'howfsc.util', 'howfsc.util.check', 'howfsc.util.insertinto', 
        'howfsc.util.loadyaml', 'howfsc.scripts', 'howfsc.scripts.gitlframes',
        'corgisim', 'corgisim.scene', 'corgisim.instrument', 'corgisim.outputs',
        'matplotlib', 'matplotlib.pyplot', 'proper'
    ]
    for module in mocks:
        sys.modules[module] = Mock()

# Set up mocks before importing
setup_mocks()

# Now we can safely import
with patch.dict('sys.modules', {
    'howfsc.util.check': Mock(
        twoD_array=Mock(side_effect=lambda arr, name, exc: None if isinstance(arr, np.ndarray) and arr.ndim == 2 else exec('raise exc("must be 2D")')),
        real_positive_scalar=Mock(side_effect=lambda val, name, exc: None if val > 0 else exec('raise exc("must be positive")')),
        positive_scalar_integer=Mock(side_effect=lambda val, name, exc: None if isinstance(val, int) and val > 0 else exec('raise exc("must be positive int")')),
        nonnegative_scalar_integer=Mock(side_effect=lambda val, name, exc: None if isinstance(val, int) and val >= 0 else exec('raise exc("must be non-negative int")'))
    )
}):
    from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
    from corgihowfsc.utils.corgisim_utils import map_wavelength_to_corgisim_bandpass

# Fixtures
@pytest.fixture
def mock_cfg():
    cfg = Mock()
    cfg.sl_list = [Mock(), Mock()]  # Need at least 2 items since code uses [1]
    cfg.sl_list[0].lam = 575e-9
    cfg.sl_list[1].lam = 660e-9  # Code uses index [1] for corgihowfsc
    return cfg

@pytest.fixture
def mock_cstrat():
    return Mock()

@pytest.fixture
def mock_hconf():
    return {'star': {'stellar_vmag': 2.5, 'stellar_type': 'G2V'}}


def test_invalid_backend(mock_cfg, mock_cstrat, mock_hconf):
    """Test invalid backend raises error"""
    with pytest.raises(ValueError, match="backend must be"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend='wrong', cor='narrowfov')


def test_invalid_cor_mode_corgihowfsc(mock_cfg, mock_cstrat, mock_hconf):
    """Test invalid cor mode raises error only for corgihowfsc backend"""
    # This should fail because corgihowfsc backend validates cor mode
    with pytest.raises(ValueError, match="corgihowfsc backend does not support cor mode"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend='corgihowfsc', cor='wrong_mode')
    
    # This should NOT fail because cgi-howfsc doesn't validate cor mode in __init__
    gitl = GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend='cgi-howfsc', cor='wrong_mode')
    assert gitl.cor == 'wrong_mode'


def test_missing_params(mock_hconf):
    """Test missing required params raise error"""
    with pytest.raises(ValueError, match="cfg and cstrat are required"):
        GitlImage(None, None, mock_hconf, cor='narrowfov')


def test_wavelength_mapping_from_cfg(cfg):
    """For each real cfg, map the middle wavelength to a CORGISIM bandpass."""
    idx = len(cfg.sl_list) // 2
    lam = cfg.sl_list[idx].lam
    band = map_wavelength_to_corgisim_bandpass(lam)
    assert band in {"1", "2", "3", "4"}, f"Unexpected band {band} for lam={lam} (idx={idx}, nlam={len(cfg.sl_list)})"

def test_cgi_needs_crop(mock_cfg, mock_cstrat, mock_hconf):
    """Test CGI backend crop validation in check_gitlframeinputs"""
    gitl = GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend='cgi-howfsc', cor='narrowfov')
    
    # The actual error happens in check_gitlframeinputs, not get_image
    with pytest.raises(TypeError, match="crop must be a 4-tuple for cgi-howfsc"):
        gitl.check_gitlframeinputs(
            dm1v=np.zeros((48,48)), 
            dm2v=np.zeros((48,48)), 
            fixedbp=np.zeros((1024,1024), dtype=bool),
            exptime=1.0,
            crop=None,  # This triggers the error
            cleanrow=1024,
            cleancol=1024
        )


def test_missing_cor_param(mock_cfg, mock_cstrat, mock_hconf):
    """Test that cor=None raises an error"""
    with pytest.raises(ValueError, match="cor mode must be provided"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend='cgi-howfsc', cor=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])