from eetc.cgi_eetc import CGIEETC

class Normalization():
    # def __init__(self):

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()


class EETCNormalization():
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, hconf):
        a, peakflux = cgi_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][j],
        )
        return a, peakflux


class CorgiNormalization():
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()