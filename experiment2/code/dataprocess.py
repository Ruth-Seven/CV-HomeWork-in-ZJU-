from config import *
import torch



class Preprocess(object):

    def __init__(self, config):
        self._config = config
        #
        self._dataset = None
        self._dataLoader = None


    def get_dataset(self):
        self._set_dataset()
        self._set_data_loader()
        return self._dataLoader


    def _set_dataset(self):
        ...


    def _set_data_loader(self):
        ...

