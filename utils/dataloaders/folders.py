from utils.dataloaders import DataLoader
import os
from utils import get_fullpath
import numpy as np


class FolderLoader(DataLoader):
    def load(self):
        self.load_articles(self.config.dataset.dataset_path, self.config.dataset.dataset_name,
                           100, 100)
        return self.articles, self.original_articles, self.vocabulary, self.frequency

    def get_content(self, filename: str, type: str = 'fake'):
        with open(filename, 'r', encoding="utf-8", errors='ignore') as document:
            return self._get_content(filename, document.read(), type)

    def load_articles(self, path, articles_directory, number_fake, number_real):
        self.articles['fake'] = []
        self.articles['real'] = []
        files_path_fake = get_fullpath(path, articles_directory, 'Fake')
        files_path_fake_titles = get_fullpath(path, articles_directory, 'Fake_titles')
        files_path_real = get_fullpath(path, articles_directory, 'Real')
        files_path_real_titles = get_fullpath(path, articles_directory, 'Real_titles')
        files_fake = np.random.choice(os.listdir(files_path_fake), number_fake)  # Get all files in the fake directory
        files_real = np.random.choice(os.listdir(files_path_real), number_real)  # Get all files in the real directory
        for file in files_fake:
            self.articles['fake'].append({
                'content': self.get_content(get_fullpath(files_path_fake, file), type='fake'),
                'title': self.get_content(get_fullpath(files_path_fake_titles, file), type='fake')
            })
        for file in files_real:
            self.articles['real'].append({
                'content': self.get_content(get_fullpath(files_path_real, file), type='real'),
                'title': self.get_content(get_fullpath(files_path_real_titles, file), type='real')
            })
