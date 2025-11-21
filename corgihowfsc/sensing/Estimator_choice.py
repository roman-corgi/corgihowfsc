import numpy as np

from corgihowfsc.sensing.Estimator import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield
class DefaultEstimator(Estimator): # Returns the estimation of the electric field directly from PWP


    def estimate_efield(self, intensities, phases,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):

        efield = estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestclip)

        return efield

# This new class allows you to focus on the EFC by directly using the electric field propagated by the optical model rather than by PWP.
# For stability reasons, the PWP is still performed according to my tests, it is just not used.

class PerfectEstimator(Estimator): # Returns the perfect electric field directly from model_efield parameter : this version work with corgisim

    def __init__(self, model_efield=None): # Initialize with the perfect electric field
        self.model_efield = model_efield

    def set_model_efield(self, model_efield): # Update the perfect E-field (call this after each howfsc_computation iteration)
        self.model_efield = model_efield

    def estimate_efield(self, intensities, phases,
                        min_good_probes=2, eestclip=np.inf, eestcondlim=0): # Return the perfect E-field directly (ignore all parameters)
        return self.model_efield
