"""
Per-mode model file registry consumed by howfsc_initialization.load_files().

Bookkeeping which hconf/probe files exist for each coronagraph
mode. Note that cstrat is not listed here: every mode's default cstrat file follows
cstrat_example_{mode}_{dark_hole}.yaml, so it is built from mode + dark_hole
directly in load_files() instead of being duplicated per mode here.

Filenames here are just the basename, not full paths. load_files() resolves DEFAULT_FILES[mode] relative to the mode (model/<mode>/) and PROBE_FILES[mode] entries relative to model/probes/.

See test/test_load_files.py for coverage of every (mode, probe_shape)
combination currently registered here.
"""

from collections import namedtuple

DEFAULT_FILES = {
    'nfov_band1': 'hconf_nfov_flat.yaml',
    'spec_band2': 'hconf_spec_band2.yaml',
    'spec_band3': 'hconf_spec_band3.yaml',
    'wfov_band4': 'hconf_wfov_band4.yaml',
    'specrot_band2': 'hconf_specrot_band2.yaml',
    'specrot_band3': 'hconf_specrot_band3.yaml',
    'wfov_band1': 'hconf_wfov_band1.yaml',
}

# 'nfov_band1' has all probe shapes populated (default, single, gaussian, and
# unmodulated_sinc) -- use it as the reference for how to fill in a mode's
# probe file list. To support a probe shape beyond these four, add a new
# field (with a None default) to ProbeFiles below; every existing mode's
# entry keeps working unchanged since unset fields default to None.
ProbeFiles = namedtuple(
    'ProbeFiles', ['default', 'single', 'gaussian', 'unmodulated_sinc'],
    defaults=[None, None, None, None],  # default value for each probe shape
)

PROBE_FILES = {
    'nfov_band1': ProbeFiles(
        default=['nfov_band1_dmrel_4_1.0e-05_cos.fits', 'nfov_band1_dmrel_4_1.0e-05_sinlr.fits', 'nfov_band1_dmrel_4_1.0e-05_sinud.fits'],
        single=['nfov_band1_dmrel_1.0e-05_act0.fits', 'nfov_band1_dmrel_1.0e-05_act1.fits', 'nfov_band1_dmrel_1.0e-05_act2.fits'],
        gaussian=['nfov_band1_dmrel_4_1.0e-05_gaussian0.fits', 'nfov_band1_dmrel_4_1.0e-05_gaussian1.fits', 'nfov_band1_dmrel_4_1.0e-05_gaussian2.fits'],
        unmodulated_sinc=['nfov_band1_dmrel_4_1.0e-05_sinc.fits', 'nfov_band1_dmrel_4_1.0e-05_sinc_shifted_right.fits', 'nfov_band1_dmrel_4_1.0e-05_sinc_shifted_diag_ur.fits'],
    ),
    'spec_band2': ProbeFiles(),
    'spec_band3': ProbeFiles(
        default=['spec_band3_dmrel_ni1e-05_sin150_rot0.fits', 'spec_band3_dmrel_ni1e-05_sin210_rot0.fits', 'spec_band3_dmrel_ni1e-05_sin90_rot0.fits'],
    ),
    'wfov_band4': ProbeFiles(
        default=['wfov_band4_dmrel_1e-5_cos_constrained.fits', 'wfov_band4_dmrel_1e-5_sinlr_constrained.fits', 'wfov_band4_dmrel_1e-5_sinud_constrained.fits'],
    ),
    'wfov_band1': ProbeFiles(
        default=['wfov_band1_dmrel_1e-5_cos.fits', 'wfov_band1_dmrel_1e-5_sinlr_constrained.fits', 'wfov_band1_dmrel_1e-5_sinud.fits']
    ),
    'specrot_band2': ProbeFiles(),
    'specrot_band3': ProbeFiles(),
}
