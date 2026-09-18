"""
Integration test for run_corgisim_nulling_gitl.py script.

Tests a complete single-iteration GITL loop with minimal configuration,
verifying that all components initialize correctly and outputs are generated.

Usage:
    pytest test_run_corgisim_nulling_gitl.py -v                    # Run all tests
    pytest test_run_corgisim_nulling_gitl.py -m "not slow"         # Skip slow tests
    pytest test_run_corgisim_nulling_gitl.py -m "not mpi"          # Skip MPI tests
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from howfsc.model.mode import CoronagraphMode
from howfsc.control.cs import ControlStrategy
from howfsc.util.loadyaml import loadyaml

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files, get_cpu_allocation
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.PerfectEstimator import PerfectEstimator
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_normalization import EETCNormalization
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.output_management import make_output_file_structure
from corgihowfsc.gitl.nulling_gitl import nulling_gitl

# Check if roman_preflight_proper is available
try:
    import roman_preflight_proper
    ROMAN_PREFLIGHT_AVAILABLE = True
except ImportError:
    ROMAN_PREFLIGHT_AVAILABLE = False

HOWFSCPATH = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

# Test configurations - minimal modes for fast testing
TEST_MODES = ['nfov_band1', 'wfov_band4']


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def minimal_config():
    """Minimal configuration for single-iteration GITL test."""
    return {
        'active_model': 'cgi-howfsc',
        'runtime': {
            'num_proper_process': 1,
            'num_jac_process': 1,
            'num_imager_worker': None,
            'use_mpi': False,
            'debug': False,
        },
        'sim_settings': {
            'loop_framework': 'corgihowfsc',
            'precomp': False,
            'output_every_iter': True,
            'niter': 1,
            'mode': 'nfov_band1',
            'dark_hole': 'both_sides',
            'probe_shape': 'default',
        },
        'crop': {
            'nrow': 128,
            'ncol': 128,
        },
        'models': {
            'cgi-howfsc': {
                'backend_type': 'cgi-howfsc',
                'normalization_type': 'eetc',
                'dmstartmap_filenames': None,
                'starting_contrast': 3.5e-4,
                'estimator': 'default',
                'lrow': 436,  # For cgi-howfsc backend (compact model)
                'lcol': 436,  # For cgi-howfsc backend (compact model)
                'corgi_overrides': {},
            }
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================

def setup_gitl_components(config, temp_output_dir, mode='nfov_band1'):
    """
    Set up all components needed for GITL loop.

    Returns tuple of (args, cfg, cstrat, hconf, imager, estimator, probes,
                     normalization, metadata, output_paths)
    """
    # Update config with test mode
    config['sim_settings']['mode'] = mode

    # Extract configuration sections
    active_model = config['active_model']
    sim_settings = config['sim_settings']
    crop_cfg = config['crop']
    model_cfg = config['models'][active_model]
    runtime = config['runtime']

    mode = sim_settings['mode']
    dark_hole = sim_settings['dark_hole']
    probe_shape = sim_settings['probe_shape']
    precomp = sim_settings['precomp']
    niter = sim_settings['niter']

    backend_type = model_cfg['backend_type']
    dmstartmap_filenames = model_cfg['dmstartmap_filenames']

    # Set up paths
    base_path = Path(temp_output_dir)
    defjacpath = os.path.join(temp_output_dir, 'jac')
    os.makedirs(defjacpath, exist_ok=True)

    # Create output structure
    fileout_path = make_output_file_structure(
        sim_settings['loop_framework'],
        backend_type,
        base_path,
        '',
        'test_gitl',
        tag='test'
    )

    # Get CPU allocation
    num_jac_process, num_imager_worker, num_proper_process = get_cpu_allocation(
        runtime['num_jac_process'],
        runtime['num_imager_worker'],
        runtime['num_proper_process'],
    )

    # Set up arguments
    args = get_args(
        niter=niter,
        mode=mode,
        dark_hole=dark_hole,
        probe_shape=probe_shape,
        precomp=precomp,
        num_process=num_jac_process,
        num_threads=1,
        fileout=fileout_path,
        jacpath=defjacpath,
        path_overrides={},
        dmstartmap_filenames=dmstartmap_filenames,
        logfile=os.path.join(os.path.dirname(fileout_path), 'test_gitl.log')
    )

    args.starting_contrast = float(model_cfg['starting_contrast'])
    args.num_imager_worker = num_imager_worker
    args.num_proper_process = num_proper_process
    args.use_mpi = False
    args.debug = runtime.get('debug', False)
    args.mpi_comm = None

    # Load model files
    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
        args, HOWFSCPATH
    )

    # Initialize optical model
    cfg = CoronagraphMode(cfgfile)
    hconf = loadyaml(hconffile, custom_exception=TypeError)
    cstrat = ControlStrategy(cstratfile)

    # Initialize probes and estimator
    probes = ProbesShapes(args.probe_shape)

    if model_cfg['estimator'] == 'perfect':
        estimator = PerfectEstimator()
        # Reduce number of probe pairs to speed up
        probefiles = {0: probefiles[0]}
        hconf['probe']['dmrel_ph_list'] = hconf['probe']['dmrel_ph_list'][:1]
    elif model_cfg['estimator'] == 'default':
        estimator = DefaultEstimator()
    else:
        raise ValueError(f"Invalid estimator: {model_cfg['estimator']}")

    # Crop parameters
    crop_params = {
        'nrow': crop_cfg['nrow'],
        'ncol': crop_cfg['ncol'],
        'lrow': model_cfg['lrow'],
        'lcol': model_cfg['lcol'],
    }

    # Corgi overrides
    corgi_overrides = model_cfg.get('corgi_overrides', {}).copy()
    corgi_overrides['output_dim'] = crop_params['nrow']

    if num_proper_process is not None:
        corgi_overrides['NCPUS'] = num_proper_process

    # Initialize imager
    imager = GitlImage(
        cfg=cfg,
        cstrat=cstrat,
        hconf=hconf,
        backend=backend_type,
        cor=mode,
        corgi_overrides=corgi_overrides
    )

    # Normalization strategy
    normalization_strategy = EETCNormalization(backend_type, corgi_overrides)

    # Metadata
    metadata = {
        "active_model": active_model,
        "backend_type": backend_type,
        "normalization_type": model_cfg['normalization_type'],
        "niter": args.niter,
        "mode": args.mode,
        "dark_hole": args.dark_hole,
        "probe_shape": args.probe_shape,
        "precomp": args.precomp,
        "num_process": args.num_process,
        "num_threads": args.num_threads,
        "num_imager_worker": args.num_imager_worker,
        "num_proper_process": args.num_proper_process,
        "use_mpi": args.use_mpi,
        "crop_params": crop_params,
        "corgi_overrides": corgi_overrides,
        "fileout": str(args.fileout),
        "jacpath": str(args.jacpath),
    }

    output_paths = {
        'fileout': fileout_path,
        'logfile': args.logfile,
        'output_dir': os.path.dirname(fileout_path),
        'modelpath': modelpath,
        'jacfile': jacfile,
        'probefiles': probefiles,
        'n2clistfiles': n2clistfiles,
        'dmstartmaps': dmstartmaps,
    }

    return (args, cfg, cstrat, hconf, imager, estimator, probes,
            normalization_strategy, crop_params, metadata, output_paths)


def verify_output_files(output_dir, niter=1):
    """
    Verify that expected output files were created.

    Returns dict with file existence flags and any missing files.
    """
    expected_files = {
        'log': 'test_gitl.log',
        'final_output': 'test_gitl.fits',
    }

    missing_files = []
    found_files = {}

    for key, filename in expected_files.items():
        filepath = os.path.join(output_dir, filename)
        exists = os.path.isfile(filepath)
        found_files[key] = exists
        if not exists:
            missing_files.append(filepath)

    return {
        'all_found': len(missing_files) == 0,
        'found_files': found_files,
        'missing_files': missing_files,
    }


# =============================================================================
# Tests
# =============================================================================

class TestSingleIterationGITL:
    """Test suite for single-iteration GITL loop."""

    @pytest.mark.slow
    @pytest.mark.skipif(not ROMAN_PREFLIGHT_AVAILABLE,
                       reason="roman_preflight_proper not available")
    def test_single_iteration_compact_model(self, minimal_config, temp_output_dir):
        """Run single GITL iteration with compact model and verify outputs."""
        # Set up all components
        (args, cfg, cstrat, hconf, imager, estimator, probes,
         normalization_strategy, crop_params, metadata, output_paths) = setup_gitl_components(
            minimal_config, temp_output_dir, mode='nfov_band1'
        )

        # Run single iteration
        nulling_gitl(
            cstrat,
            estimator,
            probes,
            normalization_strategy,
            imager,
            cfg,
            args,
            hconf,
            output_paths['modelpath'],
            output_paths['jacfile'],
            output_paths['probefiles'],
            output_paths['n2clistfiles'],
            crop_params,
            output_paths['dmstartmaps'],
            metadata,
            output_every_iter=True
        )

        # Verify outputs were created
        output_check = verify_output_files(output_paths['output_dir'], niter=1)

        assert output_check['all_found'], (
            f"Missing output files: {output_check['missing_files']}"
        )

        # Verify log file has content
        logfile = os.path.join(output_paths['output_dir'], 'test_gitl.log')
        if os.path.isfile(logfile):
            with open(logfile, 'r') as f:
                log_content = f.read()
                assert len(log_content) > 0, "Log file is empty"

        # Verify final output FITS file structure
        final_fits = os.path.join(output_paths['output_dir'], 'test_gitl.fits')
        if os.path.isfile(final_fits):
            with fits.open(final_fits) as hdul:
                assert len(hdul) > 0, "FITS file has no HDUs"

    @pytest.mark.slow
    @pytest.mark.skipif(not ROMAN_PREFLIGHT_AVAILABLE,
                       reason="roman_preflight_proper not available")
    @pytest.mark.parametrize("mode", TEST_MODES)
    def test_single_iteration_multiple_modes(self, minimal_config, temp_output_dir, mode):
        """Test single iteration works for multiple modes."""
        # Set up components for this mode
        (args, cfg, cstrat, hconf, imager, estimator, probes,
         normalization_strategy, crop_params, metadata, output_paths) = setup_gitl_components(
            minimal_config, temp_output_dir, mode=mode
        )

        # Run single iteration
        nulling_gitl(
            cstrat,
            estimator,
            probes,
            normalization_strategy,
            imager,
            cfg,
            args,
            hconf,
            output_paths['modelpath'],
            output_paths['jacfile'],
            output_paths['probefiles'],
            output_paths['n2clistfiles'],
            crop_params,
            output_paths['dmstartmaps'],
            metadata,
            output_every_iter=True
        )

        # Just verify it completed without errors
        output_check = verify_output_files(output_paths['output_dir'], niter=1)
        assert output_check['all_found'], f"Mode {mode} failed to produce outputs"

    @pytest.mark.slow
    @pytest.mark.skipif(not ROMAN_PREFLIGHT_AVAILABLE,
                       reason="roman_preflight_proper not available")
    def test_single_iteration_perfect_estimator(self, minimal_config, temp_output_dir):
        """Test single iteration with perfect estimator (faster convergence)."""
        # Modify config to use perfect estimator
        minimal_config['models']['cgi-howfsc']['estimator'] = 'perfect'

        # Set up components
        (args, cfg, cstrat, hconf, imager, estimator, probes,
         normalization_strategy, crop_params, metadata, output_paths) = setup_gitl_components(
            minimal_config, temp_output_dir, mode='nfov_band1'
        )

        # Verify perfect estimator is being used
        assert isinstance(estimator, PerfectEstimator), "Should be using PerfectEstimator"

        # Run single iteration
        nulling_gitl(
            cstrat,
            estimator,
            probes,
            normalization_strategy,
            imager,
            cfg,
            args,
            hconf,
            output_paths['modelpath'],
            output_paths['jacfile'],
            output_paths['probefiles'],
            output_paths['n2clistfiles'],
            crop_params,
            output_paths['dmstartmaps'],
            metadata,
            output_every_iter=True
        )

        # Verify outputs
        output_check = verify_output_files(output_paths['output_dir'], niter=1)
        assert output_check['all_found'], "Perfect estimator test failed to produce outputs"


class TestComponentInitialization:
    """Test that all components initialize correctly before GITL loop."""

    def test_setup_gitl_components_returns_valid_objects(self, minimal_config, temp_output_dir):
        """Verify setup_gitl_components returns properly initialized objects."""
        (args, cfg, cstrat, hconf, imager, estimator, probes,
         normalization_strategy, crop_params, metadata, output_paths) = setup_gitl_components(
            minimal_config, temp_output_dir, mode='nfov_band1'
        )

        # Verify args
        assert args.niter == 1
        assert args.mode == 'nfov_band1'
        assert args.dark_hole == 'both_sides'
        assert args.probe_shape == 'default'

        # Verify cfg
        assert isinstance(cfg, CoronagraphMode)
        assert len(cfg.sl_list) > 0, "cfg should have wavelength channels"

        # Verify cstrat
        assert isinstance(cstrat, ControlStrategy)

        # Verify hconf
        assert isinstance(hconf, dict)
        assert 'star' in hconf
        assert 'probe' in hconf

        # Verify imager
        assert isinstance(imager, GitlImage)

        # Verify estimator
        assert isinstance(estimator, (DefaultEstimator, PerfectEstimator))

        # Verify probes
        assert isinstance(probes, ProbesShapes)

        # Verify normalization
        assert isinstance(normalization_strategy, EETCNormalization)

        # Verify crop_params
        assert 'nrow' in crop_params
        assert 'ncol' in crop_params
        assert crop_params['nrow'] == 128
        assert crop_params['ncol'] == 128

        # Verify metadata
        assert 'mode' in metadata
        assert 'backend_type' in metadata
        assert metadata['mode'] == 'nfov_band1'

        # Verify output_paths
        assert 'fileout' in output_paths
        assert 'logfile' in output_paths
        assert os.path.isabs(output_paths['fileout'])

    @pytest.mark.parametrize("mode", TEST_MODES)
    def test_component_initialization_all_modes(self, minimal_config, temp_output_dir, mode):
        """Test that components initialize correctly for all test modes."""
        (args, cfg, cstrat, hconf, imager, estimator, probes,
         normalization_strategy, crop_params, metadata, output_paths) = setup_gitl_components(
            minimal_config, temp_output_dir, mode=mode
        )

        assert args.mode == mode
        assert isinstance(cfg, CoronagraphMode)
        assert isinstance(imager, GitlImage)


class TestOutputVerification:
    """Test output file verification helper functions."""

    def test_verify_output_files_with_missing_files(self, temp_output_dir):
        """verify_output_files should detect missing files."""
        # Don't create any files
        result = verify_output_files(temp_output_dir, niter=1)

        assert not result['all_found']
        assert len(result['missing_files']) > 0

    def test_verify_output_files_with_existing_files(self, temp_output_dir):
        """verify_output_files should detect existing files."""
        # Create expected files
        log_file = os.path.join(temp_output_dir, 'test_gitl.log')
        fits_file = os.path.join(temp_output_dir, 'test_gitl.fits')

        # Create dummy files
        Path(log_file).touch()
        Path(fits_file).touch()

        result = verify_output_files(temp_output_dir, niter=1)

        assert result['all_found']
        assert len(result['missing_files']) == 0
        assert result['found_files']['log']
        assert result['found_files']['final_output']


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-ra', '--tb=short', '-m', 'not slow'])
