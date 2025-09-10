from corgihowfsc.sensing.Estimate import Estimator
from howfsc.sensing.pairwise_sensing import estimate_efield
class DefaultEstimator(Estimator):
    # def __init__():

    def estimate_efield(self, intensities, phases,
                    min_good_probes=2, eestclip=np.inf, eestcondlim=0):
        efield = estimate_efield(intensities, phases,
        min_good_probes = min_good_probes, eestclip = eestclip, eestcondlim = eestclip)

        return efield