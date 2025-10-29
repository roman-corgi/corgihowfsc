from eetc import cgi_eetc
from corgihowfsc.utils.corgisim_utils import _extract_host_properties_from_hconf, calculate_mas_per_lamD
import numpy as np
from corgihowfsc.utils.corgisim_manager import CorgisimManager


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
    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None, separation_lamD=None):
        super().__init__()

        self.separation_lamD = separation_lamD

        # Initialize corgisime manager
        self.corgisim_manager = CorgisimManager(cfg, cstrat, hconf, cor, corgi_overrides=corgi_overrides)

    def calc_flux_rate(self, dm1v, dm2v, lind, exptime, gain=1):
        """
        Calculate peak flux rate for normalization using an off-axis point source (with the stellar PSF input), placed at specified separation in lambda/D. 

        Args:
            dm1v: 2D array of DM1 voltages
            dm2v: 2D array of DM2 voltages
            lind: integer, index of the wavelength slice to use
            exptime: float, exposure time
            gain: float, gain factor (default 1.0)

        Returns:
            peakflux: float, peak flux value from the off-axis PSF

        Note: the off-axis source is placed at (dx=0, dy=separation_lamD) in mas. lambda/D is calculated at the central wavelength of the bandpass.
        
        """
        # Inject off-axis source at specified separation at central wavelength
        # TODO - make sure this is the central wavelength
        mas_per_lamD = calculate_mas_per_lamD(self.corgisim_manager.cfg.sl_list[1].lam)
        
        dy = self.separation_lamD * mas_per_lamD
        dx = 0.

        image_comp_corgi = self.corgisim_manager.generate_off_axis_psf(dm1v, dm2v, dx, dy, lind=lind, exptime=exptime, gain=gain)
    
        return np.max(image_comp_corgi)

    def normalize(self, im, peakflux, exptime):
        if exptime is None or self.corgisim_manager.is_noise_free:
            exptime = 1.  # unit

        # TODO - check the normalisation for the noise case
        return im / peakflux / exptime