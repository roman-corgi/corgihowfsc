import numpy as np

from corgihowfsc.sensing.Estimator import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield

class DefaultEstimator(Estimator):
    def estimate_efield(self, intensities, phases, imager, dmlist, lam_idx, crop,
                        min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        efield = estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestclip)

        return efield
    