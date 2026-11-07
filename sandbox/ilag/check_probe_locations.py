#!/usr/bin/env python3
"""
Sanity check of DM probe locations with respect to the Lyot stop of the
HOWFSC compact model (corgihowfsc / cgi-howfsc).

Generated with Claude Opus 5.5

Usage
-----
    python check_probe_locations.py probe0.fits probe1.fits probe2.fits
    python check_probe_locations.py probe0.fits probe1.fits probe2.fits --out check.png
    python check_probe_locations.py p*.fits --cfg /path/to/howfsc_optical_model.yaml
    python check_probe_locations.py p0.fits p1.fits p2.fits --panel-order 1 2 3

Each FITS file holds one DM command (nact x nact, in volts, relative to the
current DM setting).

What is checked
---------------
The script rebuilds the exact geometric path the compact model uses:

  * howfsc.model.singlelambda.get_dmph_list / proptodm: the DM command is
    mapped to a surface with howfsc.util.dmhtoph(**registration) on a padded
    grid, then cropped with a centered insertinto() onto the pupil array.
    DM1 has z = 0, so this is exactly the pupil-plane footprint.
  * singlelambda.proptolyot: pupil -> FPM (MFT) -> Lyot (inverse MFT, chosen
    "to avoid image rotation"), and with the HLC Babinet branch the Lyot mask
    multiplies an array on the same grid as the pupil field. Pixel (r, c) of
    the DM map therefore sits on pixel (r, c) of lyot_amp.fits.

Global checks (once per run):
  1. Entrance pupil and Lyot stop have the same shape and pixels per pupil
     diameter (pdp); otherwise the shared grid assumption breaks.
  2. Centering of the Lyot stop and entrance pupil relative to the array center
     used by dmhtoph / insertinto (n // 2).
  3. DM registration: single actuators are poked across the DM, their
     footprints are located, and an affine actuator -> pixel transform is
     fitted. Its scale, orientation, parity and residuals are reported and
     compared with the registration in the YAML file. The DM footprint and
     actuator (0, 0) are drawn on the figure.

Per probe:
  * center (energy-weighted centroid of the half-maximum lobe) in array
    pixels and in units of pupil diameter D from the array center;
  * which Lyot opening it falls in, the Lyot and pupil transmission at the
    center, and the signed distance to the nearest Lyot edge;
  * fraction of the probe energy (sum of surface^2) on open Lyot pixels,
    on open pupil pixels, and on both;
  * whether the half-maximum footprint is fully clear of the Lyot edges;
  * consistency: the centroid of the command in actuator space, pushed through
    the fitted registration, must land on the measured pupil-plane center.

Outputs: a figure as PNG and PDF (one panel per probe + one panel with all probes), a text
report and a JSON file with all numbers, next to the figure.

Note: values are DM commands in volts mapped through the influence function
(no gain map), which is all that matters for location. Image orientation is
set by --origin; all reported coordinates are array (row, col) indices and do
not depend on it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import ndimage

from howfsc.util.loadyaml import loadyaml
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.insertinto import insertinto

PROBE_COLORS = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"]


# =============================================================================
# Model
# =============================================================================
def default_cfg():
    """howfsc_optical_model.yaml of corgihowfsc NFOV band 1, if installed."""
    try:
        import corgihowfsc
    except ImportError:
        return None
    return (Path(os.path.dirname(os.path.abspath(corgihowfsc.__file__)))
            / "model" / "nfov_band1" / "nfov_band1_360deg" / "howfsc_optical_model.yaml")


DM_NAME = "DM1"   # the probes are applied on DM1


def load_model(cfg_file, subband=0):
    """Load the Lyot stop, entrance pupil and DM1 registration from the YAML."""
    cfg_file = Path(cfg_file)
    cfg_dir = cfg_file.parent
    cfg = loadyaml(str(cfg_file))
    sl = cfg["sls"][subband]
    dm = cfg["dms"][DM_NAME]

    def amp(entry):
        a = np.abs(fits.getdata(cfg_dir / entry["afn"]).astype(float))
        return np.clip(a / a.max(), 0, 1)

    lyot = amp(sl["lyot"])
    epup = amp(sl["epup"])

    reg = dict(dm["registration"])
    inf_func = fits.getdata(cfg_dir / reg.pop("inffn")).astype(float)
    reg["inf_func"] = inf_func

    return dict(cfg_file=str(cfg_file), dm_name=DM_NAME, subband=subband,
                lyot=lyot, epup=epup, reg=reg, z=float(dm.get("z", 0.0)),
                pdp_lyot=float(sl["lyot"]["pdp"]), pdp_epup=float(sl["epup"]["pdp"]),
                shape=lyot.shape)


def dm_to_pupil(cmd, model):
    """Map a DM command onto the pupil/Lyot grid exactly as the model does.

    get_dmph_list computes dmhtoph on a padded (nxfresnel) grid and proptodm
    crops it back with a centered insertinto; both use the n // 2 center.
    """
    nrow, ncol = model["shape"]
    npad = 2 * max(nrow, ncol)
    surf = dmhtoph(npad, npad, np.asarray(cmd, dtype=float), **model["reg"])
    return insertinto(surf, (nrow, ncol))


# =============================================================================
# Global checks
# =============================================================================
def lobe_centroid(surf, frac=0.5):
    """Energy-weighted centroid of the main lobe (connected region >= frac*max)."""
    a = np.abs(surf)
    pk = np.unravel_index(np.argmax(a), a.shape)
    lab, _ = ndimage.label(a >= frac * a[pk])
    lobe = lab == lab[pk]
    w = np.where(lobe, surf ** 2, 0.0)
    r, c = ndimage.center_of_mass(w)
    return np.array([r, c]), lobe, pk


def registration_check(model, step=6):
    """Poke actuators across the DM and fit an affine actuator -> pixel map."""
    nact = model["reg"]["nact"]
    idx = sorted(set(list(range(0, nact, step)) + [nact - 1]))
    acts, pix = [], []
    for j in idx:
        for k in idx:
            cmd = np.zeros((nact, nact))
            cmd[j, k] = 1.0
            rc, _, _ = lobe_centroid(dm_to_pupil(cmd, model))
            acts.append([j, k])
            pix.append(rc)
    acts, pix = np.array(acts, float), np.array(pix)
    design = np.c_[acts, np.ones(len(acts))]
    coef, *_ = np.linalg.lstsq(design, pix, rcond=None)   # pix = design @ coef
    resid = pix - design @ coef
    A, b = coef[:2].T, coef[2]                            # pix = A @ [j, k] + b

    def act_to_pix(jk):
        return np.asarray(jk, float) @ A.T + b

    # Actuator axes expressed in the pixel frame
    scale_row_axis = np.linalg.norm(A[:, 0])   # moving one actuator row (index j)
    scale_col_axis = np.linalg.norm(A[:, 1])   # moving one actuator column (index k)
    # Direction (degrees, in (col, row) pixel axes) of increasing actuator column
    ang_col_axis = np.degrees(np.arctan2(A[0, 1], A[1, 1]))
    parity = "same handedness" if np.linalg.det(A) > 0 else "mirrored"

    corners_act = np.array([[-0.5, -0.5], [-0.5, nact - 0.5],
                            [nact - 0.5, nact - 0.5], [nact - 0.5, -0.5]])
    return dict(
        n_pokes=len(acts),
        A=A.tolist(), b=b.tolist(),
        px_per_act_along_act_rows=float(scale_row_axis),
        px_per_act_along_act_cols=float(scale_col_axis),
        act_col_axis_angle_deg=float(ang_col_axis),
        parity=parity,
        rms_residual_px=float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        max_residual_px=float(np.max(np.linalg.norm(resid, axis=1))),
        act00_px=act_to_pix([0, 0]).tolist(),
        footprint_px=act_to_pix(corners_act).tolist(),
        act_to_pix=act_to_pix,
    )


def global_checks(model, regchk):
    lyot, epup = model["lyot"], model["epup"]
    nrow, ncol = model["shape"]
    center = np.array([nrow // 2, ncol // 2], float)
    pdp = model["pdp_lyot"]
    checks = []

    ok = epup.shape == lyot.shape
    checks.append(("Pupil and Lyot arrays share one grid (shape)",
                   "OK" if ok else "FAIL",
                   f"epup {epup.shape}, lyot {lyot.shape}"))
    ok = np.isclose(model["pdp_epup"], model["pdp_lyot"])
    checks.append(("Same sampling (pixels per pupil diameter)",
                   "OK" if ok else "FAIL",
                   f"epup {model['pdp_epup']:.3f}, lyot {model['pdp_lyot']:.3f} px/D"))

    for name, m in [("Lyot stop", lyot), ("Entrance pupil", epup)]:
        cm = np.array(ndimage.center_of_mass(m > 0.5))
        off = cm - center
        status = "OK" if np.hypot(*off) <= 1.0 else "WARNING"
        checks.append((f"{name} centered on array center (n//2)", status,
                       f"offset (row, col) = ({off[0]:+.2f}, {off[1]:+.2f}) px "
                       f"= ({off[0] / pdp:+.4f}, {off[1] / pdp:+.4f}) D"))

    reg = model["reg"]
    ppact = [reg["ppact_cy"], reg["ppact_cx"]]
    meas = sorted([regchk["px_per_act_along_act_rows"], regchk["px_per_act_along_act_cols"]])
    ok = np.allclose(meas, sorted(ppact), rtol=0.01)
    checks.append(("DM registration: actuator pitch on the pupil grid",
                   "OK" if ok else "WARNING",
                   f"measured {meas[0]:.3f} / {meas[1]:.3f} px/act, "
                   f"YAML ppact_cy/cx {ppact[0]:.3f} / {ppact[1]:.3f}"))
    status = "OK" if regchk["max_residual_px"] < 0.5 else "WARNING"
    checks.append(("DM registration: affine fit to single-actuator pokes", status,
                   f"{regchk['n_pokes']} pokes, rms {regchk['rms_residual_px']:.3f} px, "
                   f"max {regchk['max_residual_px']:.3f} px"))
    checks.append(("DM registration: orientation", "INFO",
                   f"actuator (0,0) at pixel (row, col) = "
                   f"({regchk['act00_px'][0]:.1f}, {regchk['act00_px'][1]:.1f}); "
                   f"increasing actuator column points at "
                   f"{regchk['act_col_axis_angle_deg']:.2f} deg from +col; {regchk['parity']} "
                   f"(YAML thact = {reg['thact']:.3f}, flipx = {reg['flipx']})"))

    # Is every open Lyot pixel covered by the DM footprint?
    from matplotlib.path import Path as MplPath
    rr, cc = np.nonzero(lyot > 0.5)
    poly = MplPath(np.array(regchk["footprint_px"])[:, ::-1])   # (x=col, y=row)
    inside = poly.contains_points(np.c_[cc, rr])
    frac = inside.mean()
    checks.append(("DM footprint covers the Lyot openings",
                   "OK" if frac > 0.999 else "WARNING",
                   f"{100 * frac:.2f} % of open Lyot pixels inside the DM footprint"))

    if model["z"] != 0:
        checks.append((f"{model['dm_name']} plane", "WARNING",
                       f"z = {model['z']} m in the YAML, expected 0: the footprint shown is "
                       "a geometric projection onto the pupil grid"))
    else:
        checks.append((f"{model['dm_name']} plane", "OK",
                       "z = 0: DM1 is in the pupil plane, footprint is exact"))
    return checks


# =============================================================================
# Per-probe analysis
# =============================================================================
def analyze_probe(cmd, surf, model, regchk, openings):
    lyot, epup = model["lyot"], model["epup"]
    nrow, ncol = model["shape"]
    pdp = model["pdp_lyot"]
    ppact = 0.5 * (regchk["px_per_act_along_act_rows"] + regchk["px_per_act_along_act_cols"])
    center0 = np.array([nrow // 2, ncol // 2], float)

    open_l = lyot > 0.5
    open_p = epup > 0.5
    rc, lobe, pk = lobe_centroid(surf)
    w = surf ** 2
    wsum = w.sum()

    def interp(arr, pt):
        return float(ndimage.map_coordinates(arr.astype(float), [[pt[0]], [pt[1]]],
                                             order=1, mode="nearest")[0])

    # Signed distance to the nearest Lyot edge (>0 inside an opening)
    d_in = ndimage.distance_transform_edt(open_l)
    d_out = ndimage.distance_transform_edt(~open_l)
    d_edge = interp(d_in, rc) if interp(lyot, rc) > 0.5 else -interp(d_out, rc)

    r_hm = np.sqrt(lobe.sum() / np.pi)       # equivalent half-max radius, px
    hm_open = float(open_l[lobe].mean())     # fraction of half-max footprint on open Lyot

    lab_open, _ = openings
    pr, pc = int(round(rc[0])), int(round(rc[1]))
    opening = int(lab_open[pr, pc])          # 0 = blocked

    # Consistency: command centroid (actuator space) -> registration -> pixels
    a2 = np.asarray(cmd, float) ** 2
    jk = np.array(ndimage.center_of_mass(a2))
    pk_act = np.unravel_index(np.argmax(np.abs(cmd)), cmd.shape)
    pred = regchk["act_to_pix"](jk)
    # Compare like with like: energy centroid of the whole mapped surface
    full_cm = np.array(ndimage.center_of_mass(w))
    consistency = float(np.hypot(*(pred - full_cm)))

    if interp(lyot, rc) < 0.5:
        status = "FAIL"
        verdict = "probe center is blocked by the Lyot stop"
    elif hm_open < 1.0:
        status = "WARNING"
        verdict = (f"half-max footprint crosses a Lyot edge "
                   f"({100 * (1 - hm_open):.0f} % of it on opaque regions)")
    else:
        status = "OK"
        verdict = (f"inside Lyot opening {opening}, half-max footprint clear of edges "
                   f"by {(d_edge - r_hm) / ppact:.2f} actuators")
    if consistency > 1.0:
        status = "FAIL" if status == "FAIL" else "WARNING"
        verdict += f"; command/pupil centroid mismatch {consistency:.2f} px"

    off = rc - center0
    return dict(
        status=status, verdict=verdict,
        peak_command_V=float(cmd[pk_act]),
        peak_actuator_row_col=[int(pk_act[0]), int(pk_act[1])],
        command_centroid_act_row_col=jk.tolist(),
        center_px_row_col=rc.tolist(),
        center_offset_D_row_col=(off / pdp).tolist(),
        center_radius_D=float(np.hypot(*off) / pdp),
        lyot_opening=opening,
        lyot_transmission_at_center=interp(lyot, rc),
        pupil_transmission_at_center=interp(epup, rc),
        distance_to_lyot_edge_act=d_edge / ppact,
        distance_to_lyot_edge_D=d_edge / pdp,
        half_max_radius_act=r_hm / ppact,
        clearance_act=(d_edge - r_hm) / ppact,
        half_max_fraction_on_open_lyot=hm_open,
        energy_fraction_open_lyot=float(w[open_l].sum() / wsum),
        energy_fraction_open_pupil=float(w[open_p].sum() / wsum),
        energy_fraction_pupil_and_lyot=float(w[open_l & open_p].sum() / wsum),
        registration_consistency_px=consistency,
        _lobe_peak_px=[int(pk[0]), int(pk[1])],
    )


# =============================================================================
# I/O
# =============================================================================
def load_probes(paths, nact, labels=None):
    """Read one 2D DM command (nact x nact) per FITS file."""
    probes = []
    for p in paths:
        data = np.asarray(fits.getdata(p), dtype=float)
        if data.shape != (nact, nact):
            raise ValueError(f"{p}: expected a ({nact}, {nact}) DM command, got {data.shape}")
        probes.append(dict(source=Path(p).name, cmd=data))
    if labels:
        if len(labels) != len(probes):
            raise ValueError(f"{len(labels)} labels for {len(probes)} probes")
    else:
        labels = [f"Probe {k + 1}" for k in range(len(probes))]
    for pr, lab in zip(probes, labels):
        pr["label"] = lab
    return probes


def write_report(path, model, checks, probes):
    lines = ["Probe location sanity check", "=" * 27,
             f"Model:  {model['cfg_file']}",
             f"DM:     {model['dm_name']}   sub-band: {model['subband']}",
             f"Grid:   {model['shape'][0]} x {model['shape'][1]} px, "
             f"{model['pdp_lyot']:.3f} px per pupil diameter D",
             "Coordinates: array (row, col) indices; offsets from the array "
             "center (n//2) in units of D.", "",
             "Global checks", "-" * 13]
    for name, status, detail in checks:
        lines.append(f"[{status:7s}] {name}: {detail}")
    lines += ["", "Probes", "-" * 6]
    for p in probes:
        m = p["metrics"]
        lines += [
            f"{p['label']}  ({p['source']})",
            f"  [{m['status']}] {m['verdict']}",
            f"  peak command      {m['peak_command_V']:+.3f} V at actuator (row, col) "
            f"{tuple(m['peak_actuator_row_col'])}",
            f"  pupil-plane center (row, col) = ({m['center_px_row_col'][0]:.1f}, "
            f"{m['center_px_row_col'][1]:.1f}) px = "
            f"({m['center_offset_D_row_col'][0]:+.3f}, {m['center_offset_D_row_col'][1]:+.3f}) D, "
            f"radius {m['center_radius_D']:.3f} D",
            f"  Lyot opening {m['lyot_opening']}  (0 = blocked); transmission at center: "
            f"Lyot {m['lyot_transmission_at_center']:.2f}, pupil {m['pupil_transmission_at_center']:.2f}",
            f"  distance to nearest Lyot edge {m['distance_to_lyot_edge_act']:+.2f} act "
            f"({m['distance_to_lyot_edge_D']:+.3f} D); half-max radius "
            f"{m['half_max_radius_act']:.2f} act; clearance {m['clearance_act']:+.2f} act",
            f"  probe energy on open Lyot {100 * m['energy_fraction_open_lyot']:.1f} %, "
            f"open pupil {100 * m['energy_fraction_open_pupil']:.1f} %, both "
            f"{100 * m['energy_fraction_pupil_and_lyot']:.1f} %",
            f"  command -> pupil registration consistency {m['registration_consistency_px']:.2f} px",
            ""]
    text = "\n".join(lines)
    Path(path).write_text(text)
    return text


# =============================================================================
# Figure
# =============================================================================
def make_figure(model, probes, regchk, openings, out_file, origin="upper",
                mask_transmission=0.6, crop_margin=0.06, show=True):
    lyot = model["lyot"]
    nrow, ncol = model["shape"]
    pdp = model["pdp_lyot"]
    maps = [p["surf"] for p in probes]
    n = len(probes)

    vmax = max(np.nanmax(np.abs(m)) for m in maps)
    vmin = -vmax

    overlay = np.zeros((nrow, ncol, 4))
    overlay[..., 3] = (1.0 - mask_transmission) * (1.0 - lyot)

    # Crop: Lyot stop and DM footprint, plus a margin
    fp = np.array(regchk["footprint_px"])
    rows, cols = np.nonzero(lyot > 0.5)
    margin = crop_margin * pdp
    r0 = int(max(min(rows.min(), fp[:, 0].min()) - margin, 0))
    r1 = int(min(max(rows.max(), fp[:, 0].max()) + margin + 1, nrow))
    c0 = int(max(min(cols.min(), fp[:, 1].min()) - margin, 0))
    c1 = int(min(max(cols.max(), fp[:, 1].max()) + margin + 1, ncol))
    crop = (slice(r0, r1), slice(c0, c1))
    shift = np.array([r0, c0])

    npanel = n + 1
    ncols = 2 if npanel <= 4 else 3
    nrows = int(np.ceil(npanel / ncols))
    fig = plt.figure(figsize=(3.75 * ncols, 3.6 * nrows + 0.4))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=(0.08, 0.35),
                     share_all=True, cbar_mode="single", cbar_location="right",
                     cbar_size="4%", cbar_pad=0.12)

    fp_c = np.vstack([fp, fp[:1]]) - shift
    a00 = np.array(regchk["act00_px"]) - shift

    def decorate(ax, label_corner=False):
        ax.plot(fp_c[:, 1], fp_c[:, 0], ls="--", lw=0.8, color="0.25")
        ax.plot(a00[1], a00[0], "s", ms=4, color="0.25")
        if label_corner:
            # Put the label on the side of the marker that faces the panel center
            dx = 1 if a00[1] < (c1 - c0) / 2 else -1
            dy = 1 if a00[0] < (r1 - r0) / 2 else -1
            if origin == "upper":
                dy = -dy
            ax.annotate("act (0,0)", (a00[1], a00[0]), xytext=(5 * dx, 5 * dy),
                        textcoords="offset points", fontsize=7, color="0.25",
                        ha="left" if dx > 0 else "right",
                        va="bottom" if dy > 0 else "top")

    for ax, p in zip(grid[:n], probes):
        m = p["metrics"]
        im = ax.imshow(p["surf"][crop], cmap="bwr", vmin=vmin, vmax=vmax,
                       origin=origin, interpolation="nearest")
        ax.imshow(overlay[crop], origin=origin, interpolation="nearest")
        cr, cc = np.array(m["center_px_row_col"]) - shift
        ax.contour(np.abs(p["surf"][crop]), levels=[0.5 * np.abs(p["surf"]).max()],
                   colors="k", linewidths=0.8)
        ax.plot(cc, cr, "+", color="k", ms=8, mew=1.2)
        decorate(ax)
        ax.set_title(p["label"], fontsize=12)

    # Combined panel
    ax4 = grid[n]
    centers = np.array([p["metrics"]["center_px_row_col"] for p in probes]) - shift
    spread = np.ptp(centers, axis=0).max() if n > 1 else 0.0
    r_hm = max(p["metrics"]["half_max_radius_act"] for p in probes) \
        * regchk["px_per_act_along_act_cols"]
    zoom_half = max(0.08 * pdp, 0.5 * spread + 2.5 * r_hm)
    use_inset = zoom_half < 0.25 * pdp
    axes = [ax4]
    if use_inset:
        ax_zoom = ax4.inset_axes([0.30, 0.30, 0.40, 0.40])
        axes.append(ax_zoom)

    handles = []
    for k, p in enumerate(probes):
        color = p["color"]
        a = np.abs(p["surf"][crop])
        cr, cc = centers[k]
        for ax, ms, lw in zip(axes, (6, 10), (1.2, 1.8)):
            ax.contour(a, levels=[0.5 * a.max()], colors=[color], linewidths=lw)
            ax.plot(cc, cr, "+", color=color, ms=ms, mew=lw)
        handles.append((p["index"], Line2D([], [], color=color, lw=1.5, marker="+",
                                           ms=8, label=p["label"])))
    # Legend in input (probe) order, whatever the panel order
    handles = [h for _, h in sorted(handles, key=lambda t: t[0])]

    for ax in axes:
        ax.imshow(np.zeros((r1 - r0, c1 - c0)), cmap="bwr", vmin=vmin, vmax=vmax,
                  origin=origin, interpolation="nearest", zorder=0)
        ax.imshow(overlay[crop], origin=origin, interpolation="nearest", zorder=0.5)
    decorate(ax4, label_corner=True)

    if use_inset:
        zr, zc = centers.mean(axis=0)
        ax_zoom.set_xlim(zc - zoom_half, zc + zoom_half)
        if origin == "upper":
            ax_zoom.set_ylim(zr + zoom_half, zr - zoom_half)
        else:
            ax_zoom.set_ylim(zr - zoom_half, zr + zoom_half)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax4.indicate_inset_zoom(ax_zoom, edgecolor="black", alpha=0.8)

    # Legend below the combined panel, so it never hides the Lyot stop
    ax4.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(n, 3), fontsize=8, frameon=False, handlelength=1.5,
               columnspacing=1.0)
    ax4.set_title("All probes", fontsize=12)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in grid[npanel:]:
        ax.set_visible(False)

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_label(f"{model['dm_name']} command [V]", fontsize=12)

    fig.text(0.5, 0.0,
             "Gray: opaque Lyot stop. Dashed: DM footprint; square: actuator (0,0). "
             "Black/colored contours: half maximum.",
             ha="center", fontsize=7.5, color="0.3")
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    # Same figure as a vector PDF next to it (images inside stay rasterized)
    fig.savefig(Path(out_file).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probes", nargs="+", help="probe FITS files, one 2D DM command each")
    ap.add_argument("--cfg", default=None,
                    help="howfsc_optical_model.yaml (default: corgihowfsc NFOV band 1)")
    ap.add_argument("--subband", type=int, default=0, help="sub-band index in the YAML")
    ap.add_argument("--labels", nargs="+", default=None, help="one label per probe")
    ap.add_argument("--out", default="probe_location_check.png", help="output figure")
    ap.add_argument("--origin", choices=["upper", "lower"], default="upper",
                    help="image origin for display (upper = corgihowfsc plots, "
                         "lower = DS9); reported numbers do not depend on it")
    ap.add_argument("--mask-transmission", type=float, default=0.6,
                    help="display transmission of the opaque Lyot regions (0 = black)")
    ap.add_argument("--panel-order", nargs="+", type=int, default=None,
                    help="display order of the probe panels, 1-based; default for three "
                         "probes is 1 3 2, following the L of the Gaussian probes "
                         "(corner Probe 1 upper left, Probe 3 upper right, Probe 2 "
                         "lower left); otherwise input order; the report keeps input order")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args(argv)

    cfg = args.cfg or default_cfg()
    if cfg is None:
        ap.error("corgihowfsc is not importable: pass --cfg path/to/howfsc_optical_model.yaml")

    model = load_model(cfg, args.subband)
    probes = load_probes(args.probes, model["reg"]["nact"], args.labels)

    regchk = registration_check(model)
    checks = global_checks(model, regchk)
    openings = ndimage.label(model["lyot"] > 0.5)

    for k, p in enumerate(probes):
        p["index"] = k
        p["color"] = PROBE_COLORS[k % len(PROBE_COLORS)]
        p["surf"] = dm_to_pupil(p["cmd"], model)
        p["metrics"] = analyze_probe(p["cmd"], p["surf"], model, regchk, openings)

    out = Path(args.out)
    report = write_report(out.with_name(out.stem + "_report.txt"), model, checks, probes)
    summary = dict(
        model=dict(cfg_file=model["cfg_file"], dm=model["dm_name"], subband=model["subband"],
                   grid_shape=list(model["shape"]), px_per_D=model["pdp_lyot"]),
        global_checks=[dict(check=c, status=s, detail=d) for c, s, d in checks],
        registration={k: v for k, v in regchk.items() if k != "act_to_pix"},
        probes=[dict(label=p["label"], source=p["source"],
                     **{k: v for k, v in p["metrics"].items() if not k.startswith("_")})
                for p in probes])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(report)

    if args.no_show:
        matplotlib.use("Agg")
    # Default layout for three probes, matching the L the Gaussian probes form
    # in the combined panel:
    #   upper left  = Probe 1, the corner of the L
    #   upper right = Probe 3, the probe to the right of the corner
    #   lower left  = Probe 2, the probe below the corner
    #   lower right = all probes
    order = [0, 2, 1] if len(probes) == 3 else list(range(len(probes)))
    if args.panel_order:
        order = [i - 1 for i in args.panel_order]
        if sorted(order) != list(range(len(probes))):
            ap.error(f"--panel-order must be a permutation of 1..{len(probes)}")
    make_figure(model, [probes[i] for i in order], regchk, openings, out,
                origin=args.origin,
                mask_transmission=args.mask_transmission, show=not args.no_show)
    print(f"Figure: {out} and {out.with_suffix('.pdf')}\n"
          f"Report: {out.with_name(out.stem + '_report.txt')}\n"
          f"JSON:   {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
