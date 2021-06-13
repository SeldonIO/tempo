class InsightsWrapper:
    def __init__(self, manager):
        self.set_log_request = False
        self.set_log_response = False
        self._manager = manager

    def log(self, data):
        self._manager.log(data)

    def log_request(self):
        self.set_log_request = True

    def log_response(self):
        self.set_log_response = True
