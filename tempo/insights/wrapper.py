from ..serve.metadata import DEFAULT_INSIGHTS_TYPE, InsightsTypes


class InsightsWrapper:
    def __init__(self, manager):
        self.set_log_request = False
        self.set_log_response = False
        self._manager = manager

    def log(self, data, insights_type: InsightsTypes = DEFAULT_INSIGHTS_TYPE):
        self._manager.log(data, insights_type=insights_type)

    def log_request(self):
        self.set_log_request = True

    def log_response(self):
        self.set_log_response = True
