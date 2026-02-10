import numpy as np
import astropy.io.fits as pyfits

from howfsc.util.gitl_tools import remove_subnormals
from corgihowfsc.sensing.Probes import Probes
from howfsc.sensing.probephase import probe_ap
from howfsc.util.insertinto import insertinto

class ShapeProbes(Probes):
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
                      scalelist=[0.3, 0.3, 0.3, -0.3, -0.3, -0.3]):

        # Get probe commands
        # dm1_list, dm2
        # dmrel_list = [pyfits.getdata(probe0file),
        #               pyfits.getdata(probe1file),
        #               pyfits.getdata(probe2file),
        #               ]  # these are 1e-5 probe relative DM settings
        dmrel_list = [pyfits.getdata(probefiles[0]),
                      pyfits.getdata(probefiles[1]),
                      pyfits.getdata(probefiles[2]),
                      ]  # these are 1e-5 probe relative DM settings

        nlam = len(cfg.sl_list)
        self.ndm = 2 * len(dmrel_list) + 1
        self.croplist = [(self.lrow, self.lcol, self.nrow, self.ncol)] * (nlam * self.ndm)

        dm10 = dmstartmaps[0]
        dm20 = dmstartmaps[1]
        dm1_list = []
        dm2_list = []
        for index in range(nlam):
            # DM1 same per wavelength
            dm1_list.append(dm10)
            dm1_list.append(dm10 + scalelist[0] * dmrel_list[0])
            dm1_list.append(dm10 + scalelist[3] * dmrel_list[0])
            dm1_list.append(dm10 + scalelist[1] * dmrel_list[1])
            dm1_list.append(dm10 + scalelist[4] * dmrel_list[1])
            dm1_list.append(dm10 + scalelist[2] * dmrel_list[2])
            dm1_list.append(dm10 + scalelist[5] * dmrel_list[2])
            for j in range(self.ndm):
                # DM2 always same
                dm2_list.append(dm20)
                pass
            pass

        return dm1_list, dm2_list, dmrel_list, dm10, dm20

    def get_probe_ap(self, cfg, dm1_list, dm2_list, other=dict()):

        nlam = len(cfg.sl_list)
        nprobepair = (self.ndm - 1) // 2
        nrow = self.croplist[0][2]  # array size
        ncol = self.croplist[0][3]

        # log.info('4. Estimate complex electric fields and return fields ' +
        #          'with bad electric field maps')
        plist = []  # for model-based phase storage, chunked by lam

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
                plist.append(np.zeros((nprobepair, nrow, ncol)))
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
