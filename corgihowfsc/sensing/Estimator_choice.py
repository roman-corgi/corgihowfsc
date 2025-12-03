import numpy as np

from corgihowfsc.sensing.Estimator import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield as pairwise_estimate_efield

class DefaultEstimator(Estimator): # Returns the estimation of the electric field directly from PWP


    def estimate_efield(self, intensities, phases, imager, dmlist, lam_idx, crop,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):

        efield = pairwise_estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestcondlim)

        return efield

# This new class allows you to focus on the EFC by directly using the electric field propagated by the optical model rather than by PWP.
# For stability reasons, the PWP is still performed according to my tests, it is just not used.

class PerfectEstimator(Estimator):

    def estimate_efield(self, intensities, phases, imager, dmlist, lam_idx, crop,
                        min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        # Arguments
        imager = kwargs.get('imager')
        dmlist = kwargs.get('dmlist')
        lam_idx = kwargs.get('lam_idx')
        crop_info = kwargs.get('crop')

        if imager is None or dmlist is None or lam_idx is None:
            raise ValueError("PerfectEstimator needs 'imager', 'dmlist' and 'lam_idx'.")

        # Compute efield
        model_efield = imager.get_efield(
            dm1v=dmlist[0],
            dm2v=dmlist[1],
            lind=lam_idx,
            crop=crop_info
        )

        return model_efield
