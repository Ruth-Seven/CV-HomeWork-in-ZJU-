
import json

from pathlib import Path
import torch
import logging
from logging import Logger

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = Path('./../../../data/dataset')
test_data_path = data_path / 'test'
train_data_path = data_path / 'train'

