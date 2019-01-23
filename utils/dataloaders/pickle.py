import pickle
from utils.dataloaders import DataLoader
from utils import get_fullpath


class PickleLoader(DataLoader):
    def load(self):
        with open(get_fullpath(self.config.dataset.dataset_path), "rb") as file:
            to_load = pickle.load(file)
        return to_load["articles"], to_load["original_articles"], to_load["vocabulary"], to_load["frequency"]
