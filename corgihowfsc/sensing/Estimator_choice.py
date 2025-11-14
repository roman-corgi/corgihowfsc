import numpy as np

from corgihowfsc.sensing.Estimator import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield
class DefaultEstimator(Estimator):


    def estimate_efield(self, intensities, phases,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):

        efield = estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestclip)

        return efield


class PerfectEstimator(Estimator):
    """
    Estimator that returns the perfect (theoretical) electric field from the optical model.

    This estimator bypasses PWP estimation and retrieves the perfect electric field
    directly from the optical model computation, using the same DM settings and
    optical propagation as the optical model.

    This provides the theoretical baseline for what the electric field should be
    without any estimation error.

    Useful for:
    - Baseline performance comparisons (perfect knowledge vs PWP estimation)
    - Testing EFC without PWP biases
    - Testing the impact of estimation errors
    - Ideal performance benchmarking
    """

    def __init__(self, cfg=None, dm1_list=None, dm2_list=None, wavelength_idx=0):

        self.cfg = cfg
        self.dm1_list = dm1_list
        self.dm2_list = dm2_list
        self.wavelength_idx = wavelength_idx

    def set_dm_state(self, dm1_list, dm2_list, wavelength_idx=0):
        """
        Update the DM state for electric field computation.
        """
        self.dm1_list = dm1_list
        self.dm2_list = dm2_list
        self.wavelength_idx = wavelength_idx

    def estimate_efield(self, intensities, phases,
                        min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        """
        Return the perfect theoretical electric field from the optical model.

        Retrieves the perfect electric field by computing the optical propagation
        through the model using the current DM settings. This is the theoretical
        field that PWP uses as reference.
        """

        if self.cfg is None:
            raise ValueError("PerfectEstimator requires cfg (CoronagraphMode) to be initialized")

        if self.dm1_list is None or self.dm2_list is None:
            raise ValueError("PerfectEstimator requires DM settings (dm1_list, dm2_list) to be set")

        try:
            # Get current wavelength index
            j = self.wavelength_idx

            # Build DM measurement list [dm1, dm2] with current absolute settings
            dmlistmeas = [self.dm1_list[0], self.dm2_list[0]]

            # Compute the perfect electric field from optical model propagation
            # This mirrors the computation in gitl.py:

            # Step 1: Electric field at DM (after DM1 and DM2 influence)
            edm0 = self.cfg.sl_list[j].eprop(dmlistmeas)

            # Step 2: Electric field at Lyot plane
            ely = self.cfg.sl_list[j].proptolyot(edm0)

            # Step 3: Electric field in dark hole region
            edh0 = self.cfg.sl_list[j].proptodh(ely)

            # Step 4: Insert into full spatial array (same shape as estimated efield)
            model_efield = insertinto(edh0, phases.shape)

            return model_efield

        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(
                f"PerfectEstimator failed to compute perfect efield: {str(e)}. "
                "Ensure cfg, dm1_list, dm2_list are properly configured."
            ) from e
