from eetc import cgi_eetc

class Normalization():
    # def __init__(self):

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()


class EETCNormalization(Normalization):
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, hconf):
        a, peakflux = cgi_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][j],
        )
        return a, peakflux


class CorgiNormalization(Normalization):
    def __init__(self):
        super().__init__()

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()