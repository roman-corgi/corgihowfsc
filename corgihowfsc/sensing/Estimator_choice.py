import numpy as np

from corgihowfsc.sensing.Estimator import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield
class DefaultEstimator(Estimator):
    # def __init__():

    def estimate_efield(self, intensities, phases,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        efield = estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestclip)

        return efield

class PerfectEstimator(Estimator): # Returns the perfect electric field instead of the PWP estimate.

    def __init__(self, imager=None, cfg=None):
        self.imager = imager
        self.cfg = cfg

    def estimate_efield(self, intensities, phases,
                        min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        # Retrieve from the imager if provided
        if self.imager is not None and hasattr(self.imager, "get_perfect_efield"):
            efield_perfect = self.imager.get_perfect_efield()
            return efield_perfect

        # Retrieve from the optical model if available
        if self.cfg is not None and hasattr(self.cfg, "optical_model"):
            optical = self.cfg.optical_model
            if hasattr(optical, "get_efield"):
                return optical.get_efield()
        return phases