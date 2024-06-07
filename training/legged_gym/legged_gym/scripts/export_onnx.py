import os

import torch
from legged_gym import LEGGED_GYM_ROOT_DIR

path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1_pos_rough', 'exported')
dynamics = torch.load(os.path.join(path, '05_13_20-29-06_model_4000_nndm.pt'))
dynamics.to('cpu')
value = torch.load(os.path.join(path, '05_13_20-29-06_model_4000_ra.pt'))
value.to('cpu')

obs = torch.zeros((1, 5), dtype=torch.float32)

torch.onnx.export(dynamics, obs, os.path.join(path, 'dynamics.onnx'))
torch.onnx.export(value, obs, os.path.join(path, 'value.onnx'))
