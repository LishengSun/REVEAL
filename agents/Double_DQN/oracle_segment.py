import numpy as np
import torch
from model_segment import device

def _oracle_array(state):
    segment_length = state.shape[-1]
    first_1_index_list = np.where(state[1, :]==1)[0]
    last_0_index_list = np.where((state[1, :]==0) & (state[0, :]==1))[0]
    if len(first_1_index_list) == 0:
        first_1_index = segment_length-1
    else:
        first_1_index = first_1_index_list[0]
    if len(last_0_index_list) == 0:
        last_0_index = 0
    else:
        last_0_index = last_0_index_list[-1]
    return torch.tensor([0.]*(last_0_index + 1) + [1.]*(first_1_index - last_0_index) + [0.]*(segment_length-first_1_index - 1), device=device)/(first_1_index - last_0_index)

def oracle(s_batch, PRED_REWARD=1):
    state_cpu = s_batch.cpu().numpy()
    return torch.cat([_oracle_array(s).unsqueeze(0) for s in state_cpu])*PRED_REWARD