import torch
import math

cutoff = torch.tensor(0.25)
window_size = 11
half_size = window_size // 2
n = torch.arange(-half_size, half_size + 1, dtype=torch.float32)

sinc = 2 * cutoff * torch.sinc(2 * cutoff * n)

print("Sinc without window:", sinc)
print("Hann window:", sinc * torch.hann_window(window_size))
print("Hann window periodic:", sinc * torch.hann_window(window_size, periodic=True))
print("Hamming window:", sinc * torch.hamming_window(window_size))
print("Hamming window periodic:", sinc * torch.hamming_window(window_size, periodic=True))
print("Blackman window:", sinc * torch.blackman_window(window_size))

# Let's also test torchaudio's actual implementation by inspecting its internals
import inspect
import torchaudio.prototype.functional as F_proto
print(inspect.getsource(F_proto.sinc_impulse_response))

