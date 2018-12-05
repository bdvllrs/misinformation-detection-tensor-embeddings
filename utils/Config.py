__author__ = "Benjamin Devillers (bdvllrs)"
__credits__ = ["Benjamin Devillers (bdvllrs)"]
__license__ = "GPL"

import json
import os


class Config:
    def __init__(self, file=None, config=None, args=None, is_set=True):
        if file is not None:
            filepath = os.path.abspath(os.path.join(os.path.curdir, file))
            files = os.listdir(filepath)
            files.remove('config.default.json')
            config_filepath = os.path.abspath(os.path.join(filepath, 'config.default.json'))
            with open(config_filepath, 'r') as f:
                self.config = json.load(f)
            for file in files:
                if file[-4:] == 'json':
                    config_filepath = os.path.abspath(os.path.join(filepath, file))
                    with open(config_filepath, 'r') as f:
                        self.config = {**self.config, **json.load(f)}
        elif config is not None:
            self.config = config
        if args is not None:
            for arg, value in vars(args).items():
                if arg not in self.config.keys() or value is not None:
                    self.config[arg] = value
        self._is_set = is_set
        self.default_config = self.config.copy()

    def set(self, key, value):
        self.config[key] = value

    def reset(self, key):
        self.config[key] = self.default_config[key]

    def is_set(self, key=None):
        """
        Check if the value exists
        """
        is_set = True
        if self._is_set and key is not None:
            is_set = key in self.config.keys()
        return self._is_set and is_set

    def get(self, item):
        if self.is_set(item):
            if type(self.config[item]) == dict:
                return Config(config=self.config[item])
            return self.config[item]
        return Config(is_set=False)

    def __str__(self):
        return str(self.config)

    def __getattr__(self, item):
        return self.get(item)

    def __getitem__(self, item):
        return self.get(item)
