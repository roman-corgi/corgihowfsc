"""
Debug howfsc_initialization.load_files() against the current model tree.

This script calls load_files() directly and verifies every path it returns actually
exists on disk -- catching a broken mode/dark_hole/probe_shape combination
here instead of three calls later inside CoronagraphMode or fits.getdata().

Examples
--------
Check one mode:

    python debug_load_files.py --mode nfov_band1 --debug

Check every mode/probe_shape combination currently registered:

    python debug_load_files.py --mode all --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import corgihowfsc
from corgihowfsc.model.model_registry import DEFAULT_FILES, PROBE_FILES
from corgihowfsc.utils.howfsc_initialization import get_args, load_files

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", *sorted(DEFAULT_FILES)),
        default="nfov_band1",
        help="Mode to check, or 'all' to try every mode registered in DEFAULT_FILES.",
    )
    parser.add_argument("--dark-hole", default="both_sides", help="Dark hole variant to check.")
    parser.add_argument(
        "--probe-shape",
        default=None,
        help="Probe shape to check. Defaults to every shape populated for the mode.",
    )
    parser.add_argument(
        "--howfscpath",
        type=Path,
        default=Path(corgihowfsc.__file__).resolve().parent,
        help="Root containing the model/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "debug_load_files",
        help="Directory for the JSON summary.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser


def probe_shapes_for_mode(mode: str, requested_shape: str | None) -> list[str]:
    if requested_shape is not None:
        return [requested_shape]
    entry = PROBE_FILES.get(mode)
    if entry is None:
        return []
    return [shape for shape in entry._fields if getattr(entry, shape)]


def check_mode_files_only(mode: str, dark_hole: str, howfscpath: Path) -> dict[str, object]:
    """For modes with no registered probe shapes, still verify cfg/cstrat/hconf resolve.

    load_files() always requires a probe_shape, so this calls it with a
    placeholder and expects the failure to come specifically from probe
    resolution. Any other failure (missing cfg/cstrat/hconf, bad dark_hole)
    is a real problem and is re-raised, not swallowed.
    """
    args = get_args(mode=mode, dark_hole=dark_hole, probe_shape="default")
    try:
        load_files(args, str(howfscpath))
    except ValueError as exc:
        if "probe shape" not in str(exc).lower():
            raise
        LOG.info("[%s] cfg/cstrat/hconf resolved OK; no probe files registered yet (%s)", mode, exc)
        return {"mode": mode, "dark_hole": dark_hole, "probe_shape": None, "note": str(exc)}
    else:
        raise AssertionError(
            f"[{mode}] PROBE_FILES has no shapes registered, but load_files() succeeded anyway "
            "with probe_shape='default' -- PROBE_FILES may be stale."
        )


def check_load_files(mode: str, dark_hole: str, probe_shape: str, howfscpath: Path) -> dict[str, object]:
    LOG.info("[%s/%s/%s] calling load_files()", mode, dark_hole, probe_shape)

    args = get_args(mode=mode, dark_hole=dark_hole, probe_shape=probe_shape)
    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, _dmstartmaps = load_files(
        args, str(howfscpath)
    )

    checks = {"modelpath": modelpath, "cfgfile": cfgfile, "cstratfile": cstratfile, "hconffile": hconffile}
    if jacfile:
        checks["jacfile"] = jacfile
    checks.update({f"probefiles[{i}]": p for i, p in probefiles.items()})
    checks.update({f"n2clistfiles[{i}]": p for i, p in enumerate(n2clistfiles)})

    results = {name: {"path": path, "exists": Path(path).exists()} for name, path in checks.items()}
    for name, r in results.items():
        LOG.info("[%s] %s: %s", "OK" if r["exists"] else "MISSING", name, r["path"])

    missing = {name: r["path"] for name, r in results.items() if not r["exists"]}
    if missing:
        raise FileNotFoundError(f"[{mode}/{dark_hole}/{probe_shape}] {len(missing)} missing file(s): {missing}")

    return {"mode": mode, "dark_hole": dark_hole, "probe_shape": probe_shape, "results": results}


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted(DEFAULT_FILES) if args.mode == "all" else (args.mode,)
    failures: dict[str, str] = {}
    no_probes: list[str] = []
    summaries = []

    for mode in modes:
        shapes = probe_shapes_for_mode(mode, args.probe_shape)
        if not shapes:
            key = f"{mode}/{args.dark_hole}/(no probes registered)"
            try:
                summaries.append(check_mode_files_only(mode, args.dark_hole, args.howfscpath))
                no_probes.append(mode)
            except Exception as exc:
                LOG.exception("[%s] failed", key)
                failures[key] = f"{type(exc).__name__}: {exc}"
                if args.mode != "all":
                    raise
            continue
        for probe_shape in shapes:
            key = f"{mode}/{args.dark_hole}/{probe_shape}"
            try:
                summaries.append(check_load_files(mode, args.dark_hole, probe_shape, args.howfscpath))
            except Exception as exc:
                LOG.exception("[%s] failed", key)
                failures[key] = f"{type(exc).__name__}: {exc}"
                if args.mode != "all":
                    raise

    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"summaries": summaries, "failures": failures, "no_probes": no_probes}, stream, indent=2, default=str)

    if no_probes:
        LOG.warning("cfg/cstrat/hconf OK but no probe files registered yet: %s", no_probes)
    if failures:
        LOG.error("Failures: %s", failures)
        return 1
    if not summaries:
        LOG.error("Nothing was actually checked")
        return 1

    LOG.info("All requested load_files() checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
