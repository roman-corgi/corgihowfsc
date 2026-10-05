import os
from datetime import datetime
from pathlib import Path
import logging
import yaml
import sys

import numpy as np


def setup_logging(debug=False, logfile=None):
    """Configure root logging for the current process."""
    level = logging.DEBUG if debug else logging.INFO

    config = {
        "level": level,
        "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        "force": True,
    }

    if logfile is not None:
        config["filename"] = logfile

    logging.basicConfig(**config)


def make_output_file_structure(loop_framework, backend_type, base_path, base_corgiloop_path, final_filename, tag=None):

    if backend_type=='cgi-howfsc':
        optical_model_type = 'compact_model'
    elif backend_type=='corgihowfsc':
        optical_model_type = 'corgisim_model'
    else: raise NotImplementedError

    tag_str = f'_{tag}' if tag is not None else ''

    base_output_path = os.path.join(base_path, base_corgiloop_path, f'{loop_framework}_gitl')
    os.makedirs(base_output_path, exist_ok=True)

    current_datetime = datetime.now()
    output_folder_name = f'{current_datetime.strftime('%Y-%m-%d_%H%M%S')}_{optical_model_type}{tag_str}'

    fileout_path = os.path.join(base_output_path, output_folder_name, final_filename)
    return fileout_path


def save_run_config(args, fileout):
    """
    Save argparse Namespace (or dict) to a YAML file
    next to the provided output file.

    Args:
        args: argparse.Namespace or dict
        fileout: path to your main output file

    Returns:
        config_path (Path)
    """
    # convert args → dict safely
    cfg = vars(args).copy() if not isinstance(args, dict) else args.copy()
    
    # Runtime-only objects like MPI communicators are not YAML-serializable.
    cfg.pop("mpi_comm", None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ensure Path object
    fileout = Path(fileout)

    # build config path (same name, different extension)
    config_path = fileout.parent / "config.yml"
    # add metadata
    cfg["_meta"] = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "fileout": str(fileout),
    }

    # save yaml
    with config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return config_path


def update_yml(path, updates: dict):
    path = Path(path)
    existing = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    # pull _meta out, put updates first, then existing args, then _meta at top
    meta = existing.pop("_meta", None)
    
    merged = {}
    if meta:
        merged["_meta"] = meta
    merged.update(updates)   # active_model, backend_type, etc. come first
    merged.update(existing)  # then the args

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def plot_onboard_debug(debug_list, framelist, nlam, ndm, iteration, fileout):
    """
    Save one figure per iteration showing the onboard-processing masks for every frame.

    Frames are laid out with wavelength channels as rows and DM settings as
    columns, matching the frame index ``indj * ndm + indk``. Each panel shows
    the frame with flagged pixels coloured by source: fixed bad pixels (cyan), cosmic
    rays (fraction of the ``nframes`` raw frames in which a pixel was flagged),
    and the random bad pixels injected by ``fracbadpix``.

    Args:
        debug_list: list of per-frame debug dicts (or None) from ``_get_image_worker``
        framelist: list of the corresponding output frames
        nlam, ndm: number of wavelength channels and DM settings per channel
        iteration: iteration number, used in the filename
        fileout: path to the main output file; the figure is saved next to it.
            If None, nothing is saved.
    """
    if fileout is None:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    fig, axes = plt.subplots(nlam, ndm, figsize=(2.5 * ndm, 2.5 * nlam),
                             squeeze=False)
    for indj in range(nlam):
        for indk in range(ndm):
            ax = axes[indj, indk]
            ax.set_xticks([])
            ax.set_yticks([])
            index = indj * ndm + indk
            info = debug_list[index]
            if info is None:
                ax.text(0.5, 0.5, 'no onboard\nresult', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            bad = info['bad_pixel_map']
            cosmic = info['cosmic_ray_mask']
            cosmic_frac = cosmic.mean(axis=0)                     # fraction of frames flagged
            fixed = (bad & ~cosmic).all(axis=0)                   # bad in every frame, not from cosmics
            random_bp = info['random_bad_pixels']

            ax.imshow(np.log10(np.clip(framelist[index], 1e-10, None)),
                      origin='lower', cmap='gray')
            for mask, color in ((fixed, 'cyan'), (cosmic_frac, 'red'), (random_bp, 'lime')):
                overlay = np.zeros(mask.shape + (4,))
                overlay[..., :3] = to_rgb(color)
                overlay[..., 3] = np.where(mask > 0, np.clip(mask, 0.3, 1), 0)
                ax.imshow(overlay, origin='lower', interpolation='nearest')
            if indj == 0:
                ax.set_title(f'Probe {indk}', fontsize=8)
            if indk == 0:
                ax.set_ylabel(f'lam {indj}', fontsize=8)

    fig.suptitle(f'Iteration {iteration}: fixed (cyan), cosmic (red), random (lime)',
                 fontsize=9)
    fig.tight_layout()
    iterpath = os.path.join(os.path.dirname(fileout), f'iteration_{iteration + 1:04d}')
    os.makedirs(iterpath, exist_ok=True)
    path = os.path.join(iterpath, 'onboard_debug.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_onboard_debug(debug_list, nlam, ndm, iteration, fileout):
    """
    Save the onboard-processing masks for every frame to one FITS file.

    Each frame with a result adds four image extensions named
    ``COSMIC_{index}``, ``BADPIX_{index}``, ``NGOOD_{index}`` and
    ``RANDBP_{index}``, where ``index = indj * ndm + indk`` (wavelength channel
    ``indj``, DM setting ``indk``). The cosmic and bad-pixel masks keep their
    per-raw-frame axis, shape (nframes, ny, nx). Boolean masks are stored as uint8.
    Frames with no onboard result (e.g. noise-free mode) are skipped.

    Args:
        debug_list: list of per-frame debug dicts (or None) from ``_get_image_worker``
        nlam, ndm: number of wavelength channels and DM settings per channel
        iteration: iteration number, used in the filename
        fileout: path to the main output file; the FITS file is saved next to it.
            If None, nothing is saved.

    Returns:
        Path of the saved file, or None if nothing was saved.
    """
    if fileout is None or all(info is None for info in debug_list):
        return None

    import astropy.io.fits as pyfits

    hdr = pyfits.Header()
    hdr['ITER'] = iteration
    hdr['NLAM'] = nlam
    hdr['NDM'] = ndm
    hdul = pyfits.HDUList([pyfits.PrimaryHDU(header=hdr)])

    for index, info in enumerate(debug_list):
        if info is None:
            continue
        frame_hdr = pyfits.Header()
        frame_hdr['LIND'] = index // ndm
        frame_hdr['DMIND'] = index % ndm
        for name, key, dtype in (('COSMIC', 'cosmic_ray_mask', np.uint8),
                                 ('BADPIX', 'bad_pixel_map', np.uint8),):
            hdul.append(pyfits.ImageHDU(np.asarray(info[key]).astype(dtype),
                                        header=frame_hdr, name=f'{name}_{index}'))

    iterpath = os.path.join(os.path.dirname(fileout), f'iteration_{iteration + 1:04d}')
    os.makedirs(iterpath, exist_ok=True)
    path = os.path.join(iterpath, 'onboard_debug.fits')
    hdul.writeto(path, overwrite=True)
    return path
