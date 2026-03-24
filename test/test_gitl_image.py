"""
Pytests for GitlImage using targeted patches rather than mocking the whole
howfsc package hierarchy.
"""

import importlib
import logging
import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import corgihowfsc
from howfsc.model.mode import CoronagraphMode

log = logging.getLogger(__name__)

KNOWN_CFGS = [
    ("nfov_band1", "360deg"),
    ("nfov_band1", "half_top"),
    ("spec_band2", "both_sides"),
    ("spec_band3", "both_sides"),
    ("wfov_band4", "360deg"),
]


@pytest.fixture(params=KNOWN_CFGS, ids=lambda p: f"{p[0]}-{p[1]}")
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
    cfg.modelpath = modelpath
    return cfg


@pytest.fixture
def mock_cfg():
    cfg = Mock()
    cfg.sl_list = [Mock(), Mock()]
    cfg.sl_list[0].lam = 575e-9
    cfg.sl_list[1].lam = 660e-9
    return cfg


@pytest.fixture
def mock_cstrat():
    return Mock()


@pytest.fixture
def mock_hconf():
    return {"star": {"stellar_vmag": 2.5, "stellar_type": "G2V"}}


@pytest.fixture
def patched_modules():
    """
    Patch only the external pieces GitlImage needs, then import/reload the
    modules under test while those patches are active.
    """
    fake_defaults = {
        "crop": {
            "nrow": 153,
            "ncol": 153,
        },
        "models": {
            "cgi-howfsc": {
                "lrow": 436,
                "lcol": 436,
            }
        },
    }

    def fake_twoD_array(arr, name, exc):
        if not (isinstance(arr, np.ndarray) and arr.ndim == 2):
            raise exc("must be 2D")

    def fake_real_positive_scalar(val, name, exc):
        if not (val > 0):
            raise exc("must be positive")

    def fake_positive_scalar_integer(val, name, exc):
        if not (isinstance(val, int) and val > 0):
            raise exc("must be positive int")

    def fake_nonnegative_scalar_integer(val, name, exc):
        if not (isinstance(val, int) and val >= 0):
            raise exc("must be non-negative int")

    fake_sysmods = {
        "corgisim": Mock(),
        "corgisim.scene": Mock(),
        "corgisim.instrument": Mock(),
        "corgisim.outputs": Mock(),
        "proper": Mock(),
        "matplotlib": Mock(),
        "matplotlib.pyplot": Mock(),
    }

    with (
        patch.dict(sys.modules, fake_sysmods),
        patch("howfsc.util.loadyaml.loadyaml", return_value=fake_defaults),
        patch("howfsc.util.check.twoD_array", side_effect=fake_twoD_array),
        patch("howfsc.util.check.real_positive_scalar", side_effect=fake_real_positive_scalar),
        patch("howfsc.util.check.positive_scalar_integer", side_effect=fake_positive_scalar_integer),
        patch("howfsc.util.check.nonnegative_scalar_integer", side_effect=fake_nonnegative_scalar_integer),
    ):
        gitl_mod = importlib.import_module("corgihowfsc.utils.corgisim_gitl_frames")
        utils_mod = importlib.import_module("corgihowfsc.utils.corgisim_utils")

        gitl_mod = importlib.reload(gitl_mod)
        utils_mod = importlib.reload(utils_mod)

        yield gitl_mod, utils_mod


def test_invalid_backend(mock_cfg, mock_cstrat, mock_hconf, patched_modules):
    """Test invalid backend raises error."""
    gitl_mod, _ = patched_modules
    GitlImage = gitl_mod.GitlImage

    with pytest.raises(ValueError, match="backend must be"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend="wrong", cor="narrowfov")


def test_invalid_cor_mode_corgihowfsc(mock_cfg, mock_cstrat, mock_hconf, patched_modules):
    """Test invalid cor mode raises error only for corgihowfsc backend."""
    gitl_mod, _ = patched_modules
    GitlImage = gitl_mod.GitlImage

    with pytest.raises(ValueError, match="corgihowfsc backend does not support cor mode"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend="corgihowfsc", cor="wrong_mode")

    gitl = GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend="cgi-howfsc", cor="wrong_mode")
    assert gitl.cor == "wrong_mode"


def test_missing_params(mock_hconf, patched_modules):
    """Test missing required params raise error."""
    gitl_mod, _ = patched_modules
    GitlImage = gitl_mod.GitlImage

    with pytest.raises(ValueError, match="cfg and cstrat are required"):
        GitlImage(None, None, mock_hconf, cor="narrowfov")


def test_wavelength_mapping_from_cfg(cfg, patched_modules):
    """For each real cfg, map the middle wavelength to a CORGISIM bandpass."""
    _, utils_mod = patched_modules
    map_wavelength_to_corgisim_bandpass = utils_mod.map_wavelength_to_corgisim_bandpass

    idx = len(cfg.sl_list) // 2
    wvl = cfg.sl_list[idx].lam
    band = map_wavelength_to_corgisim_bandpass(wvl)

    wavelengths = [sl.lam for sl in cfg.sl_list]

    log.info("\n--- CFG TEST ---")
    log.info("Model path: %s", cfg.modelpath)
    log.info("Wavelength list: %s", wavelengths)
    log.info("Selected index: %d", idx)
    log.info("Selected wavelength: %.2e", wvl)
    log.info("Mapped band: %s", band)

    assert band in {"1", "2", "3", "4"}, (
        f"Unexpected band {band} for wvl={wvl} (idx={idx}, nlam={len(cfg.sl_list)})"
    )


def test_cgi_needs_crop(mock_cfg, mock_cstrat, mock_hconf, patched_modules):
    """Test CGI backend crop validation in check_gitlframeinputs."""
    gitl_mod, _ = patched_modules
    GitlImage = gitl_mod.GitlImage

    gitl = GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend="cgi-howfsc", cor="narrowfov")

    with pytest.raises(TypeError, match="crop must be a 4-tuple for cgi-howfsc"):
        gitl.check_gitlframeinputs(
            dm1v=np.zeros((48, 48)),
            dm2v=np.zeros((48, 48)),
            fixedbp=np.zeros((1024, 1024), dtype=bool),
            exptime=1.0,
            crop=None,
            cleanrow=1024,
            cleancol=1024,
        )


def test_missing_cor_param(mock_cfg, mock_cstrat, mock_hconf, patched_modules):
    """Test that cor=None raises an error."""
    gitl_mod, _ = patched_modules
    GitlImage = gitl_mod.GitlImage

    with pytest.raises(ValueError, match="cor mode must be provided"):
        GitlImage(mock_cfg, mock_cstrat, mock_hconf, backend="cgi-howfsc", cor=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-ra"])