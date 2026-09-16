from dataclasses import dataclass
import warnings
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class OnboardProcessingResult:
    """Output image and masks produced by `process_onboard_frames`."""

    image: np.ndarray
    cosmic_ray_mask: np.ndarray
    bad_pixel_map: np.ndarray
    good_frame_count: np.ndarray


def _median_filter_rows(image, size):
    """Median-filter the columns of each row using nearest-edge padding."""
    if size < 1:
        raise ValueError("median-filter size must be at least 1")

    # With an odd window this is centered on each pixel.  The implementation
    # also defines deterministic placement for an even window, with the extra
    # sample on the lower-column side.
    pad_before = size // 2
    pad_after = size - pad_before - 1
    padded = np.pad(image, ((0, 0), (pad_before, pad_after)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=size, axis=1
    )
    return np.median(windows, axis=-1)


def make_cosmic_ray_mask(
    bias_subtracted_frame_dn,
    full_well_dn,
    cosmic_filter_width=2,
    saturation_threshold=0.99,
    plateau_threshold=0.85,
):
    """Create the onboard cosmic-ray mask for one bias-subtracted frame.

    Each detector row is median-filtered to reject isolated bright pixels.  If
    a filtered value reaches ``saturation_threshold * full_well_dn``, the code
    walks toward lower column indices in the unfiltered row until it finds the
    beginning of the >= ``plateau_threshold * full_well_dn`` plateau.  The
    plateau beginning and every subsequent pixel in that row are marked bad.

    ``cosmic_filter_width`` is the literal onboard median-window width.  A
    value of 2 therefore takes the median of each pixel and its neighbor at the
    lower column index.  For two values, NumPy defines the median as their
    arithmetic mean.  This suppresses an isolated saturated pixel while
    retaining a two-pixel saturated plateau.

    Parameters
    ----------
    bias_subtracted_frame_dn : array_like
        Two-dimensional detector frame in DN after bias subtraction.
    full_well_dn : float
        Effective full-well capacity in DN.
    cosmic_filter_width : int, optional
        Width of the row-wise median filter in pixels.  Defaults to 2.
    saturation_threshold : float, optional
        Fraction of full well used to identify saturation.  Defaults to 0.99.
    plateau_threshold : float, optional
        Fraction of full well used to find the leading plateau edge.  Defaults
        to 0.85.

    Returns
    -------
    numpy.ndarray
        Boolean mask with ``True`` for pixels rejected as cosmic-contaminated.
    """
    frame = np.asarray(bias_subtracted_frame_dn)
    if frame.ndim != 2:
        raise ValueError("bias_subtracted_frame_dn must be a 2-D array")
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError("bias_subtracted_frame_dn must contain numeric values")
    if not np.isfinite(full_well_dn) or full_well_dn <= 0:
        raise ValueError("full_well_dn must be finite and greater than zero")
    if isinstance(cosmic_filter_width, bool) or not isinstance(
        cosmic_filter_width, (int, np.integer)
    ):
        raise TypeError("cosmic_filter_width must be an integer")
    if cosmic_filter_width < 1:
        raise ValueError("cosmic_filter_width must be at least 1")
    if not 0 < saturation_threshold <= 1:
        raise ValueError("saturation_threshold must be in (0, 1]")
    if not 0 < plateau_threshold <= saturation_threshold:
        raise ValueError(
            "plateau_threshold must be in (0, saturation_threshold]"
        )

    filtered = _median_filter_rows(frame, cosmic_filter_width)
    saturated_level = saturation_threshold * full_well_dn
    plateau_level = plateau_threshold * full_well_dn
    mask = np.zeros(frame.shape, dtype=bool)

    # Only visit rows that contain a candidate plateau.  If several candidates
    # occur in one row, masking from the earliest plateau covers all later ones.
    candidate_rows = np.flatnonzero(np.any(filtered >= saturated_level, axis=1))
    for row_index in candidate_rows:
        candidate_columns = np.flatnonzero(
            filtered[row_index] >= saturated_level
        )
        first_plateau = frame.shape[1]
        for column_index in candidate_columns:
            plateau_start = int(column_index)
            while (
                plateau_start > 0
                and frame[row_index, plateau_start] >= plateau_level
            ):
                plateau_start -= 1
            if frame[row_index, plateau_start] < plateau_level:
                plateau_start += 1
            first_plateau = min(first_plateau, plateau_start)

        if first_plateau < frame.shape[1]:
            mask[row_index, first_plateau:] = True

    return mask


def process_onboard_frames():
    """Apply the complete onboard treatment to a stack of detector frames.

    The processing order is:

    1. subtract detector bias from every raw DN frame;
    2. create a per-frame cosmic-ray mask;
    3. combine that mask with the fixed bad-pixel map;
    4. mean- or median-combine only the good samples;
    5. convert DN to electrons and divide by EM gain;
    6. subtract the gain-divided master dark; and
    7. divide by the flat field.

    Pixels for which every input frame is bad are returned as ``NaN``.  The
    input arrays are never modified.

    Returns
    -------
    OnboardProcessingResult
        Calibrated floating-point image, per-frame masks, and number of good
        samples contributing to each output pixel.
    """


    return OnboardProcessingResult(
        image=calibrated,
        cosmic_ray_mask=cosmic_masks,
        bad_pixel_mask=bad_masks,
        good_frame_count=good_frame_count,
    )
