from eetc import cgi_eetc

class Normalization():
    # def __init__(self):

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()


class EETCNormalization(Normalization):
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, cgi_eetc, hconf, sl_ind):
        a, peakflux = cgi_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][sl_ind],
        )
        return a, peakflux

        # implement normalize as a class method
    def normalize(self, im, peakflux, exptime):
        check.twoD_array(im, 'im', TypeError)
        check.real_positive_scalar(peakflux, 'peakflux', TypeError)
        check.real_positive_scalar(exptime, 'exptime', TypeError)

        return im/exptime/peakflux


class CorgiNormalization(Normalization):
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()