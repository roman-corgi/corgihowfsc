import numpy as np
from howfsc.util.insertinto import insertinto

def make_annulus_mask(nrows, ncols, radius_inner_lod, radius_outer_lod,
                      pixels_per_lod, xoffset=0.0, yoffset=0.0):
    cy = (nrows - 1) / 2.0 + yoffset
    cx = (ncols - 1) / 2.0 + xoffset
    yy, xx = np.ogrid[:nrows, :ncols]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    return (r >= radius_inner_lod * pixels_per_lod) & \
           (r <= radius_outer_lod * pixels_per_lod)


def get_ni(framelist, cfg, prev_exptime_list, peakfluxlist, normalization_strategy, ndm, nrow, ncol):
    """
    Compute mean dark hole intensity for unprobed images (indk=0) at each wavelength,
    using three mask definitions:
      - score : default mask from cfg
      - inner : 3–5 lambda/D annulus
      - outer : 5–9 lambda/D annulus

    Parameters
    ----------
    framelist : list
        Flat list of images with indexing [indj*ndm + indk].
    cfg : CoronagraphMode
        Config object whose sl_list entries have a dh FieldStop with pixperlod.
    ndm : int
        Number of DM states per wavelength (unprobed is indk=0).
    nrow, ncol : int
        Output image dimensions for insertinto.

    Returns
    -------
    ni_score : np.ndarray, shape (nlam,)
        Mean dark hole intensity using the default cfg mask.
    ni_inner : np.ndarray, shape (nlam,)
        Mean dark hole intensity using a 3–5 lambda/D annulus.
    ni_outer : np.ndarray, shape (nlam,)
        Mean dark hole intensity using a 5–9 lambda/D annulus.
    """
    nlam = len(cfg.sl_list)
    ni_score = np.zeros(nlam)
    ni_inner = np.zeros(nlam)
    ni_outer = np.zeros(nlam)

    for j in range(nlam):
        dh = cfg.sl_list[j].dh
        pixels_per_lod = dh.pixperlod  # from FieldStop constructor: isl['fs']['ppl']
        dh_shape = dh.e.shape
        unprobed_frame = normalization_strategy.normalize(framelist[j * ndm], peakfluxlist[j, 0], prev_exptime_list[j * ndm])

        # unprobed_frame = framelist[j * ndm]  # indk=0

        # # Default mask from cfg
        # dhcrop = insertinto(dh.e, (nrow, ncol)).astype('bool')
        # ni_score[j] = np.mean(unprobed_frame[dhcrop])

        # Scoring annulus: 3–9 lambda/D
        mask_score = make_annulus_mask(dh_shape[0], dh_shape[1],
                                       radius_inner_lod=3.0,
                                       radius_outer_lod=9.0,
                                       pixels_per_lod=pixels_per_lod)
        ni_score[j] = np.mean(unprobed_frame[insertinto(mask_score, (nrow, ncol)).astype('bool')])

        # Inner annulus: 3–5 lambda/D
        mask_inner = make_annulus_mask(dh_shape[0], dh_shape[1],
                                       radius_inner_lod=3.0,
                                       radius_outer_lod=5.0,
                                       pixels_per_lod=pixels_per_lod)
        ni_inner[j] = np.mean(unprobed_frame[insertinto(mask_inner, (nrow, ncol)).astype('bool')])

        # Outer annulus: 5–9 lambda/D
        mask_outer = make_annulus_mask(dh_shape[0], dh_shape[1],
                                       radius_inner_lod=5.0,
                                       radius_outer_lod=9.0,
                                       pixels_per_lod=pixels_per_lod)
        ni_outer[j] = np.mean(unprobed_frame[insertinto(mask_outer, (nrow, ncol)).astype('bool')])


    return np.mean(ni_score), np.mean(ni_inner), np.mean(ni_outer)

def get_perfect_efield(imager, abs_dm1, abs_dm2, croplist, log, nlam, ndm, speedup=True):
    # TODO: normalisation of the model e-field?
    # TODO: Is this the correct DM command?
    lam_inds = [nlam//2] if speedup else range(nlam)
    perfect_efields = []
    for j in lam_inds:
        perfect_efields.append(imager.get_efield(dm1v=abs_dm1, dm2v=abs_dm2, lind=j, crop=croplist[j * ndm]))

    if imager.backend == 'corgihowfsc' and speedup:
        # TODO - add a warning here for those who wants to speed up the corgisim by changing number of filters in cgisim_bandpasses
        log.info('Using corgisim model, so perfect e-field is same for all DM settings at a given wavelength')

    return perfect_efields

