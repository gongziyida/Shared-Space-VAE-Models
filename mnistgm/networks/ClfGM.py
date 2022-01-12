import torch
import torch.nn as nn

class ClfGM():
    def __init__(self, radius, var):
        self.gm_var = torch.eye(2,dtype=torch.float32) * flags.GM_var
        angles = np.pi * np.arange(8) / 4
        self.gm_locs = np.array([[flags.GM_radius * np.cos(a), flags.GM_radius * np.sin(a)] for a in angles])
        
    def ___call__(self, x):
        pass