from utils import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load(self):
        """
        Load data from list of folder
        :return:
        """
        raise NotImplementedError
