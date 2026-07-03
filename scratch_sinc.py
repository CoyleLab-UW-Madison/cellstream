import torch
import torchaudio.prototype.functional as F_proto

cutoff = torch.tensor(0.25)
ir = F_proto.sinc_impulse_response(cutoff, window_size=11)
print(ir)
