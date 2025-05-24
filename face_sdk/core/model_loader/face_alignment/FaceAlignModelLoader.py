"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import os
import logging.config

log_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'logging.conf')
log_path = os.path.abspath(log_path)

if os.path.exists(log_path):
    logging.config.fileConfig(log_path)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('sdk')


import torch
import face_sdk.models.network_def.mobilev3_pfld

from face_sdk.core.model_loader.BaseModelLoader import BaseModelLoader

from torch.nn.parallel import DataParallel
from torch.serialization import add_safe_globals

add_safe_globals([DataParallel])

class FaceAlignModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face landmark model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['img_size'] = self.meta_conf['input_width']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'], map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face landmark model!')
            return model, self.cfg
