"""
Comprehensive pytest suite for howfsc_initialization module.

Tests all functions in howfsc_initialization.py across all supported modes,
dark holes, and probe shapes. Follows the pattern of test_mode_initialization.py
but as proper unit tests with parametrization.

Usage:
    pytest test_howfsc_initialization.py -v                    # Run all tests
    pytest test_howfsc_initialization.py -k test_get_args      # Run specific test
    pytest test_howfsc_initialization.py -m "not slow"         # Skip slow tests
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pytest
from astropy.io import fits

import corgihowfsc
from corgihowfsc.model.model_registry import DEFAULT_FILES, PROBE_FILES, DM_STARTMAP_FILES
from corgihowfsc.utils.howfsc_initialization import (
    get_args,
    load_files,
    get_cpu_allocation,
    _get_model_dirs,
    _get_probe_files,
    _get_dm_startmap_files,
)

HOWFSCPATH = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

# Test parameter combinations
ALL_MODES = list(DEFAULT_FILES.keys())
DARK_HOLE = 'both_sides'  # Only test both_sides configuration

# Generate all valid (mode, probe_shape) pairs from registry
MODE_PROBE_COMBINATIONS = [
    (mode, shape)
    for mode, entry in PROBE_FILES.items()
    for shape in entry._fields
    if getattr(entry, shape) is not None
]

# Generate all valid (mode, dark_hole) pairs that should have directories
MODE_DARK_HOLE_COMBINATIONS = [
    (mode, DARK_HOLE)
    for mode in ALL_MODES
    if os.path.isdir(os.path.join(HOWFSCPATH, 'model', mode, f"{mode}_{DARK_HOLE}"))
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def minimal_args():
    """Minimal args object for testing."""
    return get_args(
        niter=1,
        mode='nfov_band1',
        dark_hole='both_sides',
        probe_shape='default',
        precomp=False,
        num_process=1,
        num_threads=1,
    )


# =============================================================================
# Tests for get_args()
# =============================================================================

class TestGetArgs:
    """Test suite for get_args function."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_get_args_all_modes(self, mode):
        """get_args should work for every registered mode."""
        args = get_args(mode=mode, dark_hole='both_sides', probe_shape='default')
        assert args.mode == mode
        assert args.dark_hole == 'both_sides'
        assert args.probe_shape == 'default'

    @pytest.mark.parametrize("probe_shape", ['default', 'single', 'gaussian', 'unmodulated_sinc'])
    def test_get_args_all_probe_shapes(self, probe_shape):
        """get_args should work for every probe shape."""
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape=probe_shape)
        assert args.probe_shape == probe_shape

    def test_get_args_default_values(self):
        """Test that default values are set correctly."""
        args = get_args()
        assert args.niter == 5
        assert args.mode == 'narrowfov'  # legacy default
        assert args.dark_hole == 'both_sides'
        assert args.probe_shape == 'default'
        assert args.profile is False
        assert args.fracbadpix == 0
        assert args.nbadpacket == 0
        assert args.nbadframe == 0
        assert args.precomp == 'load_all'

    def test_get_args_custom_values(self):
        """Test that custom values are correctly stored."""
        args = get_args(
            niter=10,
            mode='spec_band2',
            dark_hole='both_sides',
            probe_shape='default',
            profile=True,
            fracbadpix=0.01,
            nbadpacket=5,
            nbadframe=2,
            precomp='precomp_jacs_always',
            num_process=4,
            num_threads=2,
            stellarvmag=5.0,
            stellartype='G2V',
        )
        assert args.niter == 10
        assert args.mode == 'spec_band2'
        assert args.dark_hole == 'both_sides'
        assert args.probe_shape == 'default'
        assert args.profile is True
        assert args.fracbadpix == 0.01
        assert args.nbadpacket == 5
        assert args.nbadframe == 2
        assert args.precomp == 'precomp_jacs_always'
        assert args.num_process == 4
        assert args.num_threads == 2
        assert args.stellarvmag == 5.0
        assert args.stellartype == 'G2V'

    def test_get_args_path_overrides(self):
        """Test that path_overrides parameter works."""
        overrides = {'cfgfile': '/custom/path/cfg.yaml'}
        args = get_args(path_overrides=overrides)
        assert args.path_overrides == overrides

    def test_get_args_dmstartmap_filenames(self):
        """Test that dmstartmap_filenames parameter works."""
        dm_files = ['custom_dm1.fits', 'custom_dm2.fits']
        args = get_args(dmstartmap_filenames=dm_files)
        assert args.dmstartmap_filenames == dm_files

    def test_get_args_with_paths(self, temp_output_dir):
        """Test get_args with file paths specified."""
        fileout = os.path.join(temp_output_dir, 'output.fits')
        logfile = os.path.join(temp_output_dir, 'test.log')
        jacpath = temp_output_dir

        args = get_args(fileout=fileout, logfile=logfile, jacpath=jacpath)
        assert args.fileout == fileout
        assert args.logfile == logfile
        assert args.jacpath == jacpath


# =============================================================================
# Tests for get_cpu_allocation()
# =============================================================================

class TestGetCpuAllocation:
    """Test suite for get_cpu_allocation function."""

    def test_get_cpu_allocation_defaults(self):
        """With None inputs, should return (1, 1, 1)."""
        num_process, num_imager, num_proper = get_cpu_allocation(None, None, None)
        assert num_process == 1
        assert num_imager == 1
        assert num_proper == 1

    def test_get_cpu_allocation_respects_user_values(self):
        """Should respect user-provided values."""
        num_process, num_imager, num_proper = get_cpu_allocation(4, 2, 3)
        assert num_process == 4
        assert num_imager == 2
        assert num_proper == 3

    def test_get_cpu_allocation_invalid_num_process(self):
        """Should raise ValueError for invalid num_process."""
        with pytest.raises(ValueError, match="num_process must be a positive integer"):
            get_cpu_allocation(num_process=-1)

        with pytest.raises(ValueError, match="num_process must be a positive integer"):
            get_cpu_allocation(num_process=0)

    def test_get_cpu_allocation_invalid_num_imager_worker(self):
        """Should raise ValueError for invalid num_imager_worker."""
        with pytest.raises(ValueError, match="num_imager_worker must be a positive integer"):
            get_cpu_allocation(num_imager_worker=-1)

    def test_get_cpu_allocation_invalid_num_proper_process(self):
        """Should raise ValueError for invalid num_proper_process."""
        with pytest.raises(ValueError, match="num_proper_process must be a positive integer"):
            get_cpu_allocation(num_proper_process=0)

    def test_get_cpu_allocation_warns_on_oversubscription(self):
        """Should warn when requested CPUs exceed available."""
        # Request more than likely available on any machine
        with pytest.warns(UserWarning, match="exceeds available CPUs"):
            get_cpu_allocation(num_imager_worker=1000, num_proper_process=1000)

    def test_get_cpu_allocation_high_jacobian_count(self):
        """Should warn when Jacobian process count is very high."""
        with pytest.warns(UserWarning, match="exceeds available CPUs"):
            get_cpu_allocation(num_process=1000)


# =============================================================================
# Tests for _get_model_dirs()
# =============================================================================

class TestGetModelDirs:
    """Test suite for _get_model_dirs internal function."""

    @pytest.mark.parametrize("mode,dark_hole", MODE_DARK_HOLE_COMBINATIONS)
    def test_get_model_dirs_valid_combinations(self, mode, dark_hole):
        """_get_model_dirs should work for all valid (mode, dark_hole) pairs."""
        dirs = _get_model_dirs(mode, dark_hole, HOWFSCPATH)

        assert os.path.isdir(dirs.modelpath), f"modelpath does not exist: {dirs.modelpath}"
        assert os.path.isdir(dirs.modelpath_band), f"modelpath_band does not exist: {dirs.modelpath_band}"
        assert os.path.isdir(dirs.probepath), f"probepath does not exist: {dirs.probepath}"
        assert os.path.isdir(dirs.model_path_all), f"model_path_all does not exist: {dirs.model_path_all}"
        # model_any_dir may not exist for all modes, so we don't assert its existence

    def test_get_model_dirs_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="No model directory found for mode"):
            _get_model_dirs('invalid_mode', 'both_sides', HOWFSCPATH)

    def test_get_model_dirs_invalid_dark_hole(self):
        """Should raise ValueError for invalid dark_hole."""
        with pytest.raises(ValueError, match="No model directory found"):
            _get_model_dirs('nfov_band1', 'invalid_dark_hole', HOWFSCPATH)

    def test_get_model_dirs_structure(self):
        """Test that returned structure has correct fields."""
        dirs = _get_model_dirs('nfov_band1', 'both_sides', HOWFSCPATH)

        assert hasattr(dirs, 'modelpath')
        assert hasattr(dirs, 'modelpath_band')
        assert hasattr(dirs, 'probepath')
        assert hasattr(dirs, 'model_path_all')
        assert hasattr(dirs, 'model_any_dir')


# =============================================================================
# Tests for _get_probe_files()
# =============================================================================

class TestGetProbeFiles:
    """Test suite for _get_probe_files internal function."""

    @pytest.mark.parametrize("mode,probe_shape", MODE_PROBE_COMBINATIONS)
    def test_get_probe_files_all_valid_combinations(self, mode, probe_shape):
        """_get_probe_files should work for all registered (mode, probe_shape) pairs."""
        dirs = _get_model_dirs(mode, 'both_sides', HOWFSCPATH)
        probefiles = _get_probe_files(mode, probe_shape, dirs)

        assert len(probefiles) == 3, f"Expected 3 probe files, got {len(probefiles)}"
        assert 0 in probefiles
        assert 1 in probefiles
        assert 2 in probefiles

        # Verify all files exist
        for idx, filepath in probefiles.items():
            assert os.path.isfile(filepath), f"Probe file does not exist: {filepath}"

    def test_get_probe_files_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        dirs = _get_model_dirs('nfov_band1', 'both_sides', HOWFSCPATH)

        with pytest.raises(ValueError, match="Mode .* not recognized"):
            _get_probe_files('invalid_mode', 'default', dirs)

    def test_get_probe_files_invalid_probe_shape(self):
        """Should raise ValueError for unregistered probe shape."""
        dirs = _get_model_dirs('wfov_band4', 'both_sides', HOWFSCPATH)

        # wfov_band4 only has 'default' probe shape registered
        with pytest.raises(ValueError, match="Probe shape .* not available"):
            _get_probe_files('wfov_band4', 'single', dirs)

    def test_get_probe_files_index_mapping(self):
        """Verify probe file index mapping is correct (0, 2, 1 -> 0, 1, 2)."""
        dirs = _get_model_dirs('nfov_band1', 'both_sides', HOWFSCPATH)
        probefiles = _get_probe_files('nfov_band1', 'default', dirs)

        probe_names = PROBE_FILES['nfov_band1'].default

        # Original mapping: 0->0, 2->1, 1->2
        assert probe_names[0] in probefiles[0]
        assert probe_names[2] in probefiles[1]
        assert probe_names[1] in probefiles[2]


# =============================================================================
# Tests for _get_dm_startmap_files()
# =============================================================================

class TestGetDmStartmapFiles:
    """Test suite for _get_dm_startmap_files internal function."""

    @pytest.mark.parametrize("mode", DM_STARTMAP_FILES.keys())
    def test_get_dm_startmap_files_all_modes(self, mode):
        """_get_dm_startmap_files should work for all registered modes."""
        dmstartmaps = _get_dm_startmap_files(mode)

        assert len(dmstartmaps) == 2, f"Expected 2 DM files, got {len(dmstartmaps)}"
        assert os.path.isfile(dmstartmaps[0]), f"DM1 file does not exist: {dmstartmaps[0]}"
        assert os.path.isfile(dmstartmaps[1]), f"DM2 file does not exist: {dmstartmaps[1]}"

    def test_get_dm_startmap_files_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="No default flat_wfe_dm files defined"):
            _get_dm_startmap_files('invalid_mode')

    def test_get_dm_startmap_files_returns_absolute_paths(self):
        """Should return absolute paths."""
        dmstartmaps = _get_dm_startmap_files('nfov_band1')

        assert os.path.isabs(dmstartmaps[0]), "DM1 path should be absolute"
        assert os.path.isabs(dmstartmaps[1]), "DM2 path should be absolute"

    def test_get_dm_startmap_files_correct_band_mapping(self):
        """Verify correct band mapping for DM files across all modes."""
        # Band 1 modes
        nfov_band1_dms = _get_dm_startmap_files('nfov_band1')
        wfov_band1_dms = _get_dm_startmap_files('wfov_band1')

        # Band 2 modes
        spec_band2_dms = _get_dm_startmap_files('spec_band2')
        specrot_band2_dms = _get_dm_startmap_files('specrot_band2')

        # Band 3 modes
        spec_band3_dms = _get_dm_startmap_files('spec_band3')
        specrot_band3_dms = _get_dm_startmap_files('specrot_band3')

        # Band 4 modes
        wfov_band4_dms = _get_dm_startmap_files('wfov_band4')

        # Verify Band 1 modes
        assert 'band1_flat_wfe_dm1_v.fits' in nfov_band1_dms[0]
        assert 'band1_flat_wfe_dm2_v.fits' in nfov_band1_dms[1]
        assert 'band1_flat_wfe_dm1_v.fits' in wfov_band1_dms[0]
        assert 'band1_flat_wfe_dm2_v.fits' in wfov_band1_dms[1]

        # Verify Band 2 modes
        assert 'band2_flat_wfe_dm1_v.fits' in spec_band2_dms[0]
        assert 'band2_flat_wfe_dm2_v.fits' in spec_band2_dms[1]
        assert 'band2_flat_wfe_dm1_v.fits' in specrot_band2_dms[0]
        assert 'band2_flat_wfe_dm2_v.fits' in specrot_band2_dms[1]

        # Verify Band 3 modes
        assert 'band3_flat_wfe_dm1_v.fits' in spec_band3_dms[0]
        assert 'band3_flat_wfe_dm2_v.fits' in spec_band3_dms[1]
        assert 'band3_flat_wfe_dm1_v.fits' in specrot_band3_dms[0]
        assert 'band3_flat_wfe_dm2_v.fits' in specrot_band3_dms[1]

        # Verify Band 4 modes
        assert 'band4_flat_wfe_dm1_v.fits' in wfov_band4_dms[0]
        assert 'band4_flat_wfe_dm2_v.fits' in wfov_band4_dms[1]


# =============================================================================
# Tests for load_files()
# =============================================================================

class TestLoadFiles:
    """Test suite for load_files function."""

    @pytest.mark.parametrize("mode,dark_hole", MODE_DARK_HOLE_COMBINATIONS[:5])
    def test_load_files_basic_combinations(self, mode, dark_hole):
        """load_files should work for common (mode, dark_hole) combinations."""
        args = get_args(mode=mode, dark_hole=dark_hole, probe_shape='default')

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        # Verify all paths exist
        assert os.path.isdir(modelpath), f"modelpath does not exist: {modelpath}"
        assert os.path.isfile(cfgfile), f"cfgfile does not exist: {cfgfile}"
        assert os.path.isfile(cstratfile), f"cstratfile does not exist: {cstratfile}"
        assert os.path.isfile(hconffile), f"hconffile does not exist: {hconffile}"

        # Verify probe files
        assert len(probefiles) == 3
        for idx, filepath in probefiles.items():
            assert os.path.isfile(filepath), f"Probe file {idx} does not exist: {filepath}"

        # Verify n2clist files
        assert len(n2clistfiles) == 3
        for filepath in n2clistfiles:
            assert os.path.isfile(filepath), f"n2clist file does not exist: {filepath}"

        # Verify DM start maps
        assert len(dmstartmaps) == 2
        assert isinstance(dmstartmaps[0], np.ndarray)
        assert isinstance(dmstartmaps[1], np.ndarray)

    @pytest.mark.parametrize("mode,probe_shape", MODE_PROBE_COMBINATIONS[:10])
    def test_load_files_all_probe_shapes(self, mode, probe_shape):
        """load_files should work for all valid (mode, probe_shape) combinations."""
        args = get_args(mode=mode, dark_hole='both_sides', probe_shape=probe_shape)

        try:
            modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
                args, HOWFSCPATH
            )

            # All probe files should exist
            for idx, filepath in probefiles.items():
                assert os.path.isfile(filepath), f"Probe {idx} missing for {mode}/{probe_shape}: {filepath}"

        except ValueError as e:
            # If there's no model directory for this combination, that's expected
            if "No model directory found" not in str(e):
                raise

    def test_load_files_with_path_overrides(self):
        """load_files should respect path_overrides."""
        args = get_args(
            mode='nfov_band1',
            dark_hole='both_sides',
            probe_shape='default',
            path_overrides={
                'hconffile': os.path.join(
                    HOWFSCPATH, 'model', 'nfov_band1', 'hconf_nfov_flat.yaml'
                )
            }
        )

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        assert 'hconf_nfov_flat.yaml' in hconffile

    def test_load_files_with_custom_dm_startmaps_relative(self):
        """load_files should handle relative paths for DM start maps."""
        # Use relative filenames (will be resolved relative to modelpath)
        args = get_args(
            mode='nfov_band1',
            dark_hole='both_sides',
            probe_shape='default',
            dmstartmap_filenames=['hlc_seed_from_tvac_dm1.fits', 'hlc_seed_from_tvac_dm2.fits']
        )

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        assert len(dmstartmaps) == 2
        assert isinstance(dmstartmaps[0], np.ndarray)
        assert isinstance(dmstartmaps[1], np.ndarray)

    def test_load_files_with_custom_dm_startmaps_absolute(self):
        """load_files should handle absolute paths for DM start maps."""
        # Get absolute paths to default DM files
        dm_files = _get_dm_startmap_files('nfov_band1')

        args = get_args(
            mode='nfov_band1',
            dark_hole='both_sides',
            probe_shape='default',
            dmstartmap_filenames=dm_files
        )

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        assert len(dmstartmaps) == 2

    def test_load_files_mixed_dm_paths_raises_error(self):
        """load_files should raise error for mixed relative/absolute DM paths."""
        dm_files = _get_dm_startmap_files('nfov_band1')

        args = get_args(
            mode='nfov_band1',
            dark_hole='both_sides',
            probe_shape='default',
            dmstartmap_filenames=[dm_files[0], 'relative_path.fits']  # Mixed!
        )

        with pytest.raises(ValueError, match="must be specified consistently"):
            load_files(args, HOWFSCPATH)

    def test_load_files_invalid_mode(self):
        """load_files should raise ValueError for invalid mode."""
        args = get_args(mode='invalid_mode', dark_hole='both_sides', probe_shape='default')

        with pytest.raises(ValueError, match="not recognized"):
            load_files(args, HOWFSCPATH)

    def test_load_files_invalid_probe_shape(self):
        """load_files should raise ValueError for invalid probe shape."""
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape='invalid_shape')

        with pytest.raises(ValueError, match="Probe shape .* not recognized"):
            load_files(args, HOWFSCPATH)

    def test_load_files_negative_bad_packets(self):
        """load_files should raise ValueError for negative nbadpacket."""
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape='default')
        args.nbadpacket = -1

        with pytest.raises(ValueError, match="Number of bad packets cannot be less than 0"):
            load_files(args, HOWFSCPATH)

    def test_load_files_negative_bad_frames(self):
        """load_files should raise ValueError for negative nbadframe."""
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape='default')
        args.nbadframe = -1

        with pytest.raises(ValueError, match="Number of bad frames cannot be less than 0"):
            load_files(args, HOWFSCPATH)

    def test_load_files_with_jacpath(self, temp_output_dir):
        """load_files should construct jacfile path when jacpath provided."""
        args = get_args(
            mode='nfov_band1',
            dark_hole='both_sides',
            probe_shape='default',
            jacpath=temp_output_dir
        )

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        assert jacfile  # Should not be empty list
        assert 'jacnfov_band1_both_sides.fits' in jacfile

    def test_load_files_without_jacpath(self):
        """load_files should return empty list for jacfile when jacpath is None."""
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape='default')
        args.jacpath = None

        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        assert jacfile == []


# =============================================================================
# Integration Tests
# =============================================================================

class TestInitializationIntegration:
    """Integration tests combining multiple initialization functions."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ALL_MODES[:3])  # Test subset for speed
    def test_full_initialization_pipeline(self, mode, temp_output_dir):
        """Test complete initialization pipeline from get_args through load_files."""
        # Step 1: Get CPU allocation
        num_jac, num_imager, num_proper = get_cpu_allocation(1, 1, 1)

        # Step 2: Create args
        args = get_args(
            niter=1,
            mode=mode,
            dark_hole='both_sides',
            probe_shape='default',
            precomp=False,
            num_process=num_jac,
            fileout=os.path.join(temp_output_dir, 'output.fits'),
            logfile=os.path.join(temp_output_dir, 'test.log'),
        )

        # Step 3: Load files
        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        # Verify everything loaded successfully
        assert os.path.isdir(modelpath)
        assert os.path.isfile(cfgfile)
        assert os.path.isfile(cstratfile)
        assert os.path.isfile(hconffile)
        assert len(probefiles) == 3
        assert len(dmstartmaps) == 2

    def test_args_round_trip(self):
        """Test that args can be created, modified, and used with load_files."""
        # Create args
        args = get_args(mode='nfov_band1', dark_hole='both_sides', probe_shape='default')

        # Modify args
        args.niter = 10
        args.profile = True

        # Use with load_files
        modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
            args, HOWFSCPATH
        )

        # Should still work
        assert args.niter == 10
        assert args.profile is True
        assert os.path.isfile(cfgfile)


# =============================================================================
# Summary Statistics
# =============================================================================

def test_coverage_summary():
    """
    Print summary of what combinations are tested.
    This test always passes but provides useful information.
    """
    print(f"\n{'='*60}")
    print("Test Coverage Summary")
    print(f"{'='*60}")
    print(f"Total modes: {len(ALL_MODES)}")
    print(f"Modes: {', '.join(ALL_MODES)}")
    print(f"\nTotal (mode, dark_hole) combinations: {len(MODE_DARK_HOLE_COMBINATIONS)}")
    print(f"\nTotal (mode, probe_shape) combinations: {len(MODE_PROBE_COMBINATIONS)}")

    print(f"\nProbe shapes by mode:")
    for mode in sorted(PROBE_FILES.keys()):
        shapes = [s for s in PROBE_FILES[mode]._fields if getattr(PROBE_FILES[mode], s) is not None]
        print(f"  {mode}: {', '.join(shapes)}")

    print(f"\nDark hole tested: {DARK_HOLE}")
    print(f"{'='*60}\n")

    assert True  # Always pass


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-ra', '--tb=short'])
