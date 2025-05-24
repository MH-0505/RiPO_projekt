"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
from torch.nn.parallel import DataParallel
from torch.serialization import add_safe_globals

add_safe_globals([DataParallel])

import logging.config
import os
log_config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'logging.conf')
logging.config.fileConfig(os.path.abspath(log_config_path))
logger = logging.getLogger('sdk')

import torch

from face_sdk.core.model_loader.BaseModelLoader import BaseModelLoader

class FaceRecModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face recognition model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['mean'] = self.meta_conf['mean']
        self.cfg['std'] = self.meta_conf['std']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'], map_location=self.device, weights_only=False)
            if hasattr(model, 'module'):
                model = model.module  # Extract the actual model from DataParallel wrapper
            model = model.to('cpu')  # Ensure the model is on CPU
            model.eval()
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face recognition model!')
            return model, self.cfg
