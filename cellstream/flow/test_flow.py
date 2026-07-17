import torch
import numpy as np
import time
from .analytic import phase_velocity
from .piv import binned_piv_velocity

def generate_spiral_wave(size=100, T=50, omega=0.5):
    x = torch.arange(size) - size // 2
    y = torch.arange(size) - size // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    theta = torch.atan2(Y, X)
    t = torch.arange(T).view(T, 1, 1)
    
    # Spiral phase: theta - r + omega * t
    r = torch.sqrt(X**2 + Y**2)
    phase = theta - r/10 + omega * t
    
    # Wrap to [-pi, pi]
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    return phase

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    phase = generate_spiral_wave(size=128, T=50, omega=0.5).to(device)
    
    print("Testing Analytic Phase Velocity...")
    t0 = time.time()
    v_a, speed_a = phase_velocity(phase, smooth_sigma=1.0, device=device)
    t1 = time.time()
    print(f"Analytic Time: {t1 - t0:.4f}s")
    print(f"Velocity shape: {v_a.shape}, Speed shape: {speed_a.shape}")
    
    print("\nTesting Binned PIV Velocity...")
    t0 = time.time()
    v_piv, coords = binned_piv_velocity(phase, num_bins=8, device=device)
    t1 = time.time()
    print(f"PIV Time: {t1 - t0:.4f}s")
    print(f"Velocity shape: {v_piv.shape}")
    
    print("\nTesting Binned PIV Velocity (Upsampled)...")
    t0 = time.time()
    v_piv_up = binned_piv_velocity(phase, num_bins=8, device=device, upsample=True)
    t1 = time.time()
    print(f"PIV (Upsampled) Time: {t1 - t0:.4f}s")
    print(f"Velocity shape: {v_piv_up.shape}")
    
    print("\nTests finished without errors.")
