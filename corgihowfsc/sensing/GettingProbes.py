import numpy as np
import astropy.io.fits as pyfits

from howfsc.util.gitl_tools import remove_subnormals
from corgihowfsc.sensing.Probes import Probes
from howfsc.sensing.probephase import probe_ap
from howfsc.util.insertinto import insertinto

class ProbesShapes(Probes):
    def __init__(self,
                 probe_type,
                 nrow=153,
                 ncol=153,
                 lrow=436,
                 lcol=436):
        super().__init__(probe_type)

        self.nrow = nrow
        self.ncol = ncol
        self.lrow = lrow
        self.lcol = lcol

    def get_dm_probes(self, cfg, probefiles, dmstartmaps,
                      scalelist=None):
        """
        Build DM1 and DM2 command lists for a probed HOWFSC iteration.

        For each wavelength, the DM1 list contains one unprobed command followed
        by a positive/negative pair for each probe, giving ndm = 2*nprobepair + 1
        entries per wavelength. DM2 is held fixed at dm20 for all entries.

        Parameters
        ----------
        cfg : CoronagraphMode
            Optical model; used to determine the number of wavelengths (nlam).
        probefiles : list of str
            Paths to FITS files containing the relative DM probe commands.
            The number of probes (nprobepair) is inferred from the length of
            this list.
        dmstartmaps : list of ndarray
            Two-element list [dm10, dm20] giving the absolute starting DM
            commands in volts.
        scalelist : list of float, optional
            Scale factors applied to the probe commands. Must have length
            2 * nprobepair, where the first half are the positive-probe scales
            and the second half are the negative-probe scales. Defaults to
            [0.3] * nprobepair + [-0.3] * nprobepair.

        Returns
        -------
        dm1_list : list of ndarray
            Absolute DM1 commands for all wavelengths and DM settings,
            length nlam * ndm.
        dm2_list : list of ndarray
            Absolute DM2 commands for all wavelengths and DM settings,
            length nlam * ndm. All entries are dm20.
        dmrel_list : list of ndarray
            Relative probe DM commands loaded from probefiles, length nprobepair.
        dm10 : ndarray
            Starting absolute DM1 command (48x48 array, volts).
        dm20 : ndarray
            Starting absolute DM2 command (48x48 array, volts).

        Raises
        ------
        ValueError
            If scalelist is provided but its length is not 2 * nprobepair.
        """
        dmrel_list = [pyfits.getdata(f) for f in probefiles]
        nprobepair = len(dmrel_list)

        if scalelist is None:
            scalelist = [0.3] * nprobepair + [-0.3] * nprobepair

        if len(scalelist) != 2 * nprobepair:
            raise ValueError(
                f"scalelist length ({len(scalelist)}) must be 2 * number of probes ({2 * nprobepair})"
            )

        nlam = len(cfg.sl_list)
        self.ndm = 2 * nprobepair + 1
        self.croplist = [(self.lrow, self.lcol, self.nrow, self.ncol)] * (nlam * self.ndm)

        dm10 = dmstartmaps[0]
        dm20 = dmstartmaps[1]
        dm1_list = []
        dm2_list = []
        for index in range(nlam):
            dm1_list.append(dm10)  # unprobed
            for i, dmrel in enumerate(dmrel_list):
                dm1_list.append(dm10 + scalelist[i] * dmrel)  # positive probe
                dm1_list.append(dm10 + scalelist[i + nprobepair] * dmrel)  # negative probe
            for j in range(self.ndm):
                dm2_list.append(dm20)

        return dm1_list, dm2_list, dmrel_list, dm10, dm20

    def get_probe_ap(self, cfg, dm1_list, dm2_list, other=dict()):

        nlam = len(cfg.sl_list)
        nprobepair = (self.ndm - 1) // 2
        nrow = self.croplist[0][2]  # array size
        ncol = self.croplist[0][3]

        # log.info('4. Estimate complex electric fields and return fields ' +
        #          'with bad electric field maps')
        plist = [np.zeros((nprobepair, nrow, ncol)) for _ in range(nlam)] # for model-based phase storage, chunked by lam

        # This is a catch-all dictionary for HOWFSC information we want to export
        # from the call, but don't strictly need to set up the next iteration.
        # Will contain things like electric-field estimates, etc.

        for n in range(nprobepair):
            # log.info('Probe pair %d of %d', n + 1, nprobepair)
            # Extract phases from model
            # element zero is unprobed, not used here
            # data collection will do plus then minus
            for j in range(nlam):
                # other[j] = dict()
                # plist.append(np.zeros((nprobepair, nrow, ncol)))
                # log.info('Wavelength %d of %d', j + 1, nlam)
                # log.info('Get probe phase from model and DM settings')
                _, tmpph = probe_ap(cfg,
                                    dm1_list[j * self.ndm + 1 + 2 * n],  # positive
                                    dm1_list[j * self.ndm + 2 + 2 * n],  # negative
                                    dm2_list[j * self.ndm],
                                    j)

                plist[j][n, :, :] = insertinto(tmpph, (nrow, ncol))

                # Save the probe phases for later
                key_n = 'probe_ph' + str(n)
                other[j][key_n] = np.squeeze(plist[j][n, :, :])
                pass

            pass

        return plist, other
