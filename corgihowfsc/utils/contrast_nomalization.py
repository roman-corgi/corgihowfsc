from eetc import cgi_eetc
from corgihowfsc.utils.corgisim_utils import _extract_host_properties_from_hconf, calculate_mas_per_lamD
import numpy as np
from corgihowfsc.utils.corgisim_manager import CorgisimManager
from howfsc.util import check


class Normalization():
    # def __init__(self):

    def calc_flux_rate(self, hconf):
        raise NotImplementedError()


class EETCNormalization(Normalization):
<<<<<<< feature/debugging_outputs
    def __init__(self, backend, corgi_overrides):
=======
    """
    Normalization strategy using the CGI Exposure Time Calculator (EETC).

    Computes peak flux rates from the EETC sequence list defined in hconf,
    and normalizes detector images to contrast units by dividing by exposure
    time and peak flux.
    """

    def __init__(self, backend, corgi_overrides):
        """
        Parameters
        ----------
        backend : str
            Optical model backend, either 'corgihowfsc' or 'cgi-howfsc'.
            Used to determine whether the image is already a rate (noise-free
            corgihowfsc) or needs exposure time division.
        corgi_overrides : dict
            CorgiSim override parameters. Must contain 'is_noise_free' (bool)
            when backend is 'corgihowfsc'.
        """
>>>>>>> main
        super().__init__()
        self.backend = backend
        self.corgi_overrides = corgi_overrides

    def calc_flux_rate(self, get_cgi_eetc, hconf, sl_ind, dm1v, dm2v, gain=1):
        """
        Calculate peak flux rate using the EETC sequence list.

        Looks up the sequence name for the given wavelength index from
        hconf['hardware']['sequence_list'] and queries the EETC for the
        corresponding flux rate. dm1v, dm2v, and gain are accepted for
        interface compatibility with other normalization strategies but are
        not used.

        Parameters
        ----------
        get_cgi_eetc : CGIEETC
            Instantiated EETC object used to compute flux rates.
        hconf : dict
            Hardware configuration dict. Must contain
            hconf['hardware']['sequence_list'] with one entry per wavelength.
        sl_ind : int
            Wavelength channel index into hconf['hardware']['sequence_list'].
        dm1v : np.ndarray
            DM1 voltage map (unused, present for interface compatibility).
        dm2v : np.ndarray
            DM2 voltage map (unused, present for interface compatibility).
        gain : float, optional
            EM gain (unused, present for interface compatibility). Default 1.

        Returns
        -------
        a : float
            Flux rate.
        peakflux : float
            Peak flux rate in counts/s or photons/s.
        """
        a, peakflux = get_cgi_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][sl_ind],
        )
        return a, peakflux

    def normalize(self, im, peakflux, exptime):
        """
        Normalize a detector image to NI units.

        Divides the image by exposure time and peak flux. If the backend is
        'corgihowfsc' in noise-free mode, the image is already a rate so
        exptime is set to 1 before division.

        Parameters
        ----------
        im : np.ndarray, shape (nrow, ncol)
            Detector image in counts (or counts/s if noise-free).
        peakflux : float
            Peak flux rate in counts/s or photons/s, as returned by
            calc_flux_rate.
        exptime : float
            Exposure time in seconds. Must be > 0.

        Returns
        -------
        np.ndarray
            Normalized image in NI units (im / exptime / peakflux).
        """
        check.twoD_array(im, 'im', TypeError)
        check.real_positive_scalar(peakflux, 'peakflux', TypeError)
        check.real_positive_scalar(exptime, 'exptime', TypeError)

<<<<<<< feature/debugging_outputs
        # If the imager is the noise-free option the image is already a rate
        if self.backend == 'corgihowfsc' and self.corgi_overrides['is_noise_free']:
            exptime = 1
        return im/exptime/peakflux
=======
        if self.backend == 'corgihowfsc' and self.corgi_overrides['is_noise_free']:
            exptime = 1

        return im / exptime / peakflux
>>>>>>> main


class CorgiNormalization(Normalization):
    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None, separation_lamD=None, exptime_norm=1):
        super().__init__()

        self.separation_lamD = separation_lamD


        # Initialize corgisime manager
        self.corgisim_manager = CorgisimManager(cfg, cstrat, hconf, cor, corgi_overrides=corgi_overrides)

        if self.corgisim_manager.is_noise_free:
            self.exptime_norm = 1
        else:
            self.exptime_norm = exptime_norm


    def calc_flux_rate(self, get_cgi_eetc, hconf, sl_ind, dm1v, dm2v, gain=1):
        """
        Calculate peak flux rate for normalization using an off-axis point source (with the stellar PSF input), placed at specified separation in lambda/D. 

        Args:
            dm1v: 2D array of DM1 voltages
            dm2v: 2D array of DM2 voltages
            lind: integer, index of the wavelength slice to use
            exptime: float, exposure time [s]
            gain: float, gain factor (default 1.0)

        Returns:
            peakflux: float, peak flux value from the off-axis PSF in [counts/s] or [photons/s]

        Note: the off-axis source is placed at (dx=0, dy=separation_lamD) in mas. lambda/D is calculated at the central wavelength of the bandpass.
        
        """
        # Inject off-axis source at specified separation at central wavelength
        # TODO - make sure this is the central wavelength
        mas_per_lamD = calculate_mas_per_lamD(self.corgisim_manager.cfg.sl_list[1].lam)

        # if exptime is None or self.corgisim_manager.is_noise_free:
        #     exptime = 1.  # unit

        dy = self.separation_lamD * mas_per_lamD
        dx = 0.

        image_comp_corgi = self.corgisim_manager.generate_off_axis_psf(dm1v,
                                                                       dm2v,
                                                                       dx,
                                                                       dy,
                                                                       lind=sl_ind,
                                                                       exptime=self.exptime_norm,
                                                                       gain=gain)
        if (np.nanmax(image_comp_corgi) > 89610) and (not self.corgisim_manager.is_noise_free): #10300:
            print("**** WARNING: off-axis PSF saturated ****")

        peakflux = np.nanmax(image_comp_corgi) / self.exptime_norm

        return image_comp_corgi/self.exptime_norm, peakflux

    def normalize(self, im, peakflux, exptime):
        """
        Args:
            im: 2D array image in counts
            peakflux: peakflux for image in ph/s or counts/s
            exptime: exposure time of im in [s]

        Returns:

        """
        if exptime is None or self.corgisim_manager.is_noise_free:
            exptime = 1.  # unit

        # TODO - check the normalisation for the noise case
        return im / exptime / peakflux


class CorgiNormalizationOnAxis(CorgiNormalization):
    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None, exptime_norm=1):
        super().__init__(cfg, cstrat, hconf, cor=cor, corgi_overrides=corgi_overrides, separation_lamD=None, exptime_norm=exptime_norm)

    def calc_flux_rate(self, get_cgi_eetc, hconf, sl_ind, dm1v, dm2v, gain=1):
        """
        Calculate peak flux rate for normalization using an off-axis point source (with the stellar PSF input), placed at specified separation in lambda/D.

        Args:
            dm1v: 2D array of DM1 voltages
            dm2v: 2D array of DM2 voltages
            lind: integer, index of the wavelength slice to use
            exptime: float, exposure time [s]
            gain: float, gain factor (default 1.0)

        Returns:
            peakflux: float, peak flux value from the off-axis PSF in [counts/s] or [photons/s]

        Note: the off-axis source is placed at (dx=0, dy=separation_lamD) in mas. lambda/D is calculated at the central wavelength of the bandpass.

        """

        image_comp_corgi = self.corgisim_manager.generate_on_axis_psf(dm1v,
                                                                       dm2v,
                                                                       lind=sl_ind,
                                                                       exptime=self.exptime_norm,
                                                                       gain=gain)
        if (np.nanmax(image_comp_corgi) > 89610) and (not self.corgisim_manager.is_noise_free):  # 10300:
            print("**** WARNING: off-axis PSF saturated ****")

        peakflux = np.nanmax(image_comp_corgi) / self.exptime_norm

        return image_comp_corgi / self.exptime_norm, peakflux
