

class Probes:
    def __init__(self,
                 probe_type):
        self.probe_type = probe_type

    def get_dm_probes(self):
        raise NotImplementedError()

    def get_probe_ap(self, cfg, dm1_list, dm2_list):
        raise NotImplementedError()