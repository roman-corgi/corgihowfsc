"""
Pytests for howfsc_initialization.load_files() against the real model tree.

Exercises every (mode, probe_shape) combination currently registered in
PROBE_FILES, verifying load_files() resolves every returned path to a real,
existing file. Modes with no probe shapes registered yet are checked
separately: load_files() always raises on probe resolution for them, but
cfg/cstrat/hconf should still resolve correctly before that point.
"""

import os

import corgihowfsc
import pytest

from corgihowfsc.model.model_registry import DEFAULT_FILES, PROBE_FILES
from corgihowfsc.utils.howfsc_initialization import (
    get_args,
    load_files,
)

HOWFSCPATH = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

MODE_PROBE_SHAPE_CASES = [
    (mode, shape)
    for mode, entry in PROBE_FILES.items()
    for shape in entry._fields
    if getattr(entry, shape)
]

MODES_WITHOUT_PROBES = [
    mode
    for mode, entry in PROBE_FILES.items()
    if not any(getattr(entry, shape) for shape in entry._fields)
]


@pytest.mark.parametrize("mode,probe_shape", MODE_PROBE_SHAPE_CASES)
def test_load_files_resolves_all_paths(mode, probe_shape):
    args = get_args(mode=mode, dark_hole="both_sides", probe_shape=probe_shape)
    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(
        args, HOWFSCPATH
    )

    checks = {"modelpath": modelpath, "cfgfile": cfgfile, "cstratfile": cstratfile, "hconffile": hconffile}
    checks.update({f"probefiles[{idx}]": p for idx, p in probefiles.items()})
    checks.update({f"n2clistfiles[{idx}]": p for idx, p in enumerate(n2clistfiles)})

    missing = {name: p for name, p in checks.items() if not os.path.exists(p)}
    assert not missing, f"missing files for mode={mode!r} probe_shape={probe_shape!r}: {missing}"
    assert len(dmstartmaps) == 2


@pytest.mark.parametrize("mode", MODES_WITHOUT_PROBES)
def test_load_files_config_resolves_even_without_probes(mode):
    """cfg/cstrat/hconf should resolve even for modes with no probes yet.

    load_files() always raises once it reaches probe resolution for these
    modes (no shapes are registered in PROBE_FILES), so this asserts the
    failure is specifically about the probe shape -- not about a missing
    cfg/cstrat/hconf file, which would indicate DEFAULT_FILES itself is wrong
    for this mode.
    """
    assert mode in DEFAULT_FILES, f"{mode} has no DEFAULT_FILES entry"

    args = get_args(mode=mode, dark_hole="both_sides", probe_shape="default")
    with pytest.raises(ValueError, match="Probe shape"):
        load_files(args, HOWFSCPATH)
