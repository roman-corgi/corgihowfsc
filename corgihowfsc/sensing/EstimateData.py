class EstimateData:
    def __init__(self,
                 estimator_type):
        self.estimator_type = estimator_type

    def get_estimate_data(self, cfg, dm1_list, dm2_list):
        raise NotImplementedError()
