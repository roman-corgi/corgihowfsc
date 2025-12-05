
import numpy as np

class Estimator():
    """
    Abstract estimator base class
    """
    # def __init__():

    def estimate_efield(self, intensities, phases, imager, dmlist, lam_idx, crop,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        """
        Estimates the electric field.
        Base arguments support Pair-Wise Probing method and perfect estimator method.

        Parameters
        ----------
        intensities : np.ndarray
            A 3D array of real-valued image data.
            The first element [0,:,:] is the initial intensity, and subsequent arrays are the
            positive and negative phase perturbations; ie 2N+1 arrays for N phase perturbations.

        phases : np.ndarray
            a 3d array of phase data corresponding to the propagated
            perturbations through the optical system, in radians

        min_good_probes : int, optional
            An integer >= 2 giving the number of probe intensity measurements that must be good
            at a pixel in order for the estimation to be done at that pixel.
            Given the number of unknowns in the matrix solve, 2 good measurements is mathematical
            minimum that can be accepted. Note that 'good' in this context means not in badpixels
            as True AND does not produce an estimate with negative intensity. No other data quality
            metric is applied. Defaults to 2.

        eestclip : float, optional
            If the incoherent is less than -coherent*eestclip for an
            e-field array element, that element will be marked as
            bad. This permits incoherent estimates to be negative
            (due to e.g. read noise) but caps how negative the
            incoherent can be and how large the coherent can be.  In
            particular, this mitigates spikes like the one seen in
            PFR 218555.  Scalar value >= 0.  Setting to infinity
            disables this check.  Defaults to np.inf.

        eestcondlim : float, optional
            The minimum condition number (ratio of smallest singular value to largest in the
            least-square solve) which can be accepted as a valid estimate. As the matrix is N x 2
            for some N >= 2, there will only ever be two singular values. A poorly-conditioned
            solution is close to rank 1, which suggests the data is degenerate and not able to
            estimate independent real and imaginary parts of the data. >= 0.
            Setting to 0 disables this check. Defaults to 0.

        imager : object
            Simulation object (e.g., GitlImage) capable of generating electric fields
            or images based on the optical model. Required for Perfect Estimator.

        dmlist : list
            List of current DM command arrays [DM1, DM2] used to set the state of
            the optical model. Required for Perfect Estimator.

        lam_idx : int
            Index of the current wavelength channel being processed. Required for Perfect Estimator.

        crop : tuple
            4-tuple of (lower row, lower col, number of rows,
            number of columns), indicating where in a clean frame a PSF is taken.
            All are integers; the first two must be >= 0 and the second two must be > 0.
            Required for Perfect Estimator.

        Returns
        -------
        efield : np.ndarray
            A 2D array of complex electric field, with bad estimates as NaNs.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented by a subclass.
        """
        raise NotImplementedError("The estimate_efield method must be implemented by the subclass.")

