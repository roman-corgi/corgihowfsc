import numpy as np
import astropy.io.fits as pyfits

from howfsc.util.gitl_tools import remove_subnormals
from corgihowfsc.sensing.DefaultProbes import DefaultProbes


class SingleProbes(DefaultProbes):
    """
    Implementation of single actuator probes.
    Differs from DefaultProbes only in how the probe maps are loaded/generated so we can call it with DefaultProbes
    """

    def get_dm_probes(self, cfg, probefiles,
                      scalelist=[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]):

        # Get probe commands
        # Load single actuator maps directly from FITS files provided in probefiles (see the notebook)
        # Probefiles is expected to be a list or dict where indices 0, 1, 2 correspond to the Center, Neighbor, and Diagonal actuators respectively.
        dmrel_list = [pyfits.getdata(probefiles[0]),
                      pyfits.getdata(probefiles[1]),
                      pyfits.getdata(probefiles[2]),
                      ]

        nlam = len(cfg.sl_list)
        self.ndm = 2 * len(dmrel_list) + 1
        self.croplist = [(self.lrow, self.lcol, self.nrow, self.ncol)] * (nlam * self.ndm)

        dm10_init = cfg.initmaps[0]
        dm20_init = cfg.initmaps[1]
        dm10_cons = cfg.dmlist[0].dmvobj.constrain_dm(dm10_init)
        dm10 = remove_subnormals(dm10_cons)
        dm20_cons = cfg.dmlist[1].dmvobj.constrain_dm(dm20_init)
        dm20 = remove_subnormals(dm20_cons)

        dm1_list = []
        dm2_list = []
        for index in range(nlam):
            # DM1 same per wavelength
            dm1_list.append(dm10)

            # Apply probes: Center Actuator
            dm1_list.append(dm10 + scalelist[0] * dmrel_list[0])
            dm1_list.append(dm10 + scalelist[3] * dmrel_list[0])

            # Apply probes: Neighbor Actuator
            dm1_list.append(dm10 + scalelist[1] * dmrel_list[1])
            dm1_list.append(dm10 + scalelist[4] * dmrel_list[1])

            # Apply probes: Diagonal Actuator
            dm1_list.append(dm10 + scalelist[2] * dmrel_list[2])
            dm1_list.append(dm10 + scalelist[5] * dmrel_list[2])

            for j in range(self.ndm):
                # DM2 always same
                dm2_list.append(dm20)
                pass
            pass

        return dm1_list, dm2_list, dmrel_list, dm10, dm20