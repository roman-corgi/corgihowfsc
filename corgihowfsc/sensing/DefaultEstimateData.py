import numpy as np
import astropy.io.fits as pyfits


from corgihowfsc.sensing.EstimateData import EstimateData


class DefaultEstimateData(EstimateData):
    def __init__(self,
                 estimator_type):
        super().__init__(estimator_type)

    def get_estimate_data(self, cfg, dm1_list, dm2_list):

        # new framelist
        framelist = []
        for indj, sl in enumerate(cfg.sl_list):
            crop = croplist[indj]
            _, peakflux = cgi_eetc.calc_flux_rate(
                sequence_name=hconf['hardware']['sequence_list'][indj],
            )
            for indk in range(ndm):
                dmlist = [dm1_list[indj * ndm + indk],
                          dm2_list[indj * ndm + indk]]
                f = sim_gitlframe(cfg,
                                  dmlist,
                                  cstrat.fixedbp,
                                  peakflux,
                                  prev_exptime_list[indj * ndm + indk],
                                  crop,
                                  indj,
                                  cleanrow=hconf['excam']['cleanrow'],
                                  cleancol=hconf['excam']['cleancol'])
                framelist.append(f)
                pass
            pass

        return framelist