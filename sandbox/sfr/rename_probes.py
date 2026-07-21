import re
import os
from pathlib import Path


# def rename_dmrel_fits(filename: str) -> str:
#     """
#     Rename FITS files to the canonical format: MODE_dmrel_<everything else>
#
#     Rules:
#     - narrowfov -> nfov
#     - mode (nfov/wfov/spec) is always the first component
#     - dmrel always follows the mode
#
#     Args:
#         filename: Original filename (basename, with or without .fits)
#
#     Returns:
#         Renamed filename in canonical format
#     """
#     stem = filename.removesuffix('.fits')
#     suffix = '.fits' if filename.endswith('.fits') else ''
#
#     MODES = ('nfov', 'wfov', 'spec')
#
#     # print(stem)
#
#     # Normalize narrowfov -> nfov
#     stem = stem.replace('narrowfov', 'nfov')
#
#     # Already in correct format: MODE_dmrel_...
#     for mode in MODES:
#         if stem.startswith(f'{mode}_dmrel_'):
#             return f'{stem}{suffix}'  # was: return filename
#
#     # Pattern: MODE_dm_dmrel_... (e.g. nfov_dm_dmrel_4_1.0e-05_cos)
#     for mode in MODES:
#         m = re.fullmatch(rf'({mode})_dm_dmrel_(.*)', stem)
#         if m:
#             rest = m.group(2)
#             return f'{mode}_dmrel_{rest}{suffix}'
#
#     # Pattern: dmrel_MODE_... (e.g. dmrel_nfov_band1_..., dmrel_spec_band3_...)
#     m = re.fullmatch(r'dmrel_(' + '|'.join(MODES) + r')_(.*)', stem)
#     if m:
#         mode, rest = m.group(1), m.group(2)
#         return f'{mode}_dmrel_{rest}{suffix}'
#
#     # Pattern: MODE_dmrel_... without leading dm_ (e.g. wfov_dmrel_1e-5_...)
#     for mode in MODES:
#         m = re.fullmatch(rf'({mode})_dmrel_(.*)', stem)
#         if m:
#             rest = m.group(2)
#             return f'{mode}_dmrel_{rest}{suffix}'
#
#     raise ValueError(f"Could not parse filename: {filename!r}")
#
#
def rename_files_in_dir(directory: str | Path, dry_run: bool = True) -> list[tuple[str, str]]:
    """
    Rename all matching .fits files in a directory.

    Args:
        directory: Path to directory containing files
        dry_run: If True, print renames without executing. If False, actually rename.

    Returns:
        List of (old_name, new_name) tuples for all files that would be/were renamed.
    """
    directory = Path(directory)
    renames = []

    for filepath in sorted(directory.glob('*.fits')):
        old_name = filepath.name
        try:
            new_name = rename_dmrel_fits(old_name)
        except ValueError as e:
            print(f"  SKIP (unrecognized): {old_name}")
            continue

        if old_name != new_name:
            renames.append((old_name, new_name))
            if dry_run:
                print(f"  {old_name}\n    -> {new_name}")
            else:
                filepath.rename(directory / new_name)
                print(f"  Renamed: {old_name} -> {new_name}")

    if not renames:
        print("  No files need renaming.")
    return renames
#

  # dmrel_nfov_band1_360deg_ni1e-05_sin150_rot0.fits
  #   -> nfov_dmrel_band1_360deg_ni1e-05_sin150_rot0.fits
  # dmrel_nfov_band1_360deg_ni1e-05_sin210_rot90.fits
  #   -> nfov_dmrel_band1_360deg_ni1e-05_sin210_rot90.fits
  # dmrel_nfov_band1_360deg_ni1e-05_sin90_rot0.fits
  #   -> nfov_dmrel_band1_360deg_ni1e-05_sin90_rot0.fits
  # dmrel_spec_band3_both_sides_ni1e-05_sin150_rot0.fits
  #   -> spec_dmrel_band3_both_sides_ni1e-05_sin150_rot0.fits
  # dmrel_spec_band3_both_sides_ni1e-05_sin210_rot0.fits
  #   -> spec_dmrel_band3_both_sides_ni1e-05_sin210_rot0.fits
  # dmrel_spec_band3_both_sides_ni1e-05_sin90_rot0.fits
  #   -> spec_dmrel_band3_both_sides_ni1e-05_sin90_rot0.fits
  # dmrel_wfov_band4_360deg_ni1e-05_sin150_rot0.fits
  #   -> wfov_dmrel_band4_360deg_ni1e-05_sin150_rot0.fits
  # dmrel_wfov_band4_360deg_ni1e-05_sin210_rot90.fits
  #   -> wfov_dmrel_band4_360deg_ni1e-05_sin210_rot90.fits
  # dmrel_wfov_band4_360deg_ni1e-05_sin90_rot0.fits
  #   -> wfov_dmrel_band4_360deg_ni1e-05_sin90_rot0.fits
  # narrowfov_dmrel_1.0e-05_act0.fits
  #   -> nfov_dmrel_1.0e-05_act0.fits
  # narrowfov_dmrel_1.0e-05_act1.fits
  #   -> nfov_dmrel_1.0e-05_act1.fits
  # narrowfov_dmrel_1.0e-05_act2.fits
  #   -> nfov_dmrel_1.0e-05_act2.fits
  # narrowfov_dmrel_1.0e-05_three.fits
  #   -> nfov_dmrel_1.0e-05_three.fits
  # nfov_dm_dmrel_4_1.0e-05_cos.fits
  #   -> nfov_dmrel_4_1.0e-05_cos.fits
  # nfov_dm_dmrel_4_1.0e-05_gaussian0.fits
  #   -> nfov_dmrel_4_1.0e-05_gaussian0.fits
  # nfov_dm_dmrel_4_1.0e-05_gaussian1.fits
  #   -> nfov_dmrel_4_1.0e-05_gaussian1.fits
  # nfov_dm_dmrel_4_1.0e-05_gaussian2.fits
  #   -> nfov_dmrel_4_1.0e-05_gaussian2.fits
  # nfov_dm_dmrel_4_1.0e-05_sinc.fits
  #   -> nfov_dmrel_4_1.0e-05_sinc.fits
  # nfov_dm_dmrel_4_1.0e-05_sinc_shifted_diag_ur.fits
  #   -> nfov_dmrel_4_1.0e-05_sinc_shifted_diag_ur.fits
  # nfov_dm_dmrel_4_1.0e-05_sinc_shifted_right.fits
  #   -> nfov_dmrel_4_1.0e-05_sinc_shifted_right.fits
  # nfov_dm_dmrel_4_1.0e-05_sinlr.fits
  #   -> nfov_dmrel_4_1.0e-05_sinlr.fits
  # nfov_dm_dmrel_4_1.0e-05_sinud.fits
  #   -> nfov_dmrel_4_1.0e-05_sinud.fits


def rename_dmrel_fits(filename: str) -> str:
    """
    Rename FITS files to the canonical format: MODE_dmrel_<everything else>

    Rules:
    - narrowfov -> nfov
    - mode (nfov/wfov/spec) is always the first component
    - dmrel always follows the mode
    - 360deg_ and both_sides_ are removed from the rest

    Args:
        filename: Original filename (basename, with or without .fits)

    Returns:
        Renamed filename in canonical format
    """
    stem = filename.removesuffix('.fits')
    suffix = '.fits' if filename.endswith('.fits') else ''

    MODES = ('nfov', 'wfov', 'spec')

    # Normalize narrowfov -> nfov
    stem = stem.replace('narrowfov', 'nfov')

    # Already in correct format: MODE_dmrel_...
    for mode in MODES:
        if stem.startswith(f'{mode}_dmrel_'):
            rest = stem[len(f'{mode}_dmrel_'):]
            rest = _strip_location_tokens(rest)
            return f'{mode}_dmrel_{rest}{suffix}'

    # Pattern: MODE_dm_dmrel_... (e.g. nfov_dm_dmrel_4_1.0e-05_cos)
    for mode in MODES:
        m = re.fullmatch(rf'({mode})_dm_dmrel_(.*)', stem)
        if m:
            rest = _strip_location_tokens(m.group(2))
            return f'{mode}_dmrel_{rest}{suffix}'

    # Pattern: dmrel_MODE_... (e.g. dmrel_nfov_band1_..., dmrel_spec_band3_...)
    m = re.fullmatch(r'dmrel_(' + '|'.join(MODES) + r')_(.*)', stem)
    if m:
        mode, rest = m.group(1), m.group(2)
        rest = _strip_location_tokens(rest)
        return f'{mode}_dmrel_{rest}{suffix}'

    # Pattern: MODE_dmrel_... without leading dm_ (e.g. wfov_dmrel_1e-5_...)
    for mode in MODES:
        m = re.fullmatch(rf'({mode})_dmrel_(.*)', stem)
        if m:
            rest = _strip_location_tokens(m.group(2))
            return f'{mode}_dmrel_{rest}{suffix}'

    raise ValueError(f"Could not parse filename: {filename!r}")


def _strip_location_tokens(rest: str) -> str:
    """Remove 360deg_ and both_sides_ tokens from the 'rest' part of a filename."""
    rest = rest.replace('360deg_', '')
    rest = rest.replace('both_sides_', '')
    return rest

if __name__ == '__main__':
    path2 = r'C:\Users\sredmond\Documents\github_repos\roman-corgi-repos\corgihowfsc\corgihowfsc\model\probes'
    rename_files_in_dir(path2, False)

