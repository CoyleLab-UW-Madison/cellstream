import torch
import numpy as np

# Test imports
print("Testing imports...")
try:
    from cellstream.phase import winding_number
    from cellstream.phase import generate_phase_features
    from cellstream.phase import phase_velocity, compute_ftle
    print("Imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    exit(1)

# Test execution
print("\nTesting generate_phase_features execution...")
try:
    # Create dummy phase data (T=30, Y=64, X=64)
    T, Y, X = 30, 64, 64
    phase_tensor = torch.rand((T, Y, X), dtype=torch.float32) * 2 * np.pi - np.pi
    
    # Run the feature generator
    features = generate_phase_features(
        phase_tensor, 
        device='cpu', 
        ftle_integration_time=5, 
        smooth_sigma=1.0, 
        defect_window_size=3
    )
    
    print("\nGenerated features:")
    for key, val in features.items():
        print(f" - {key}: shape {val.shape}, dtype {val.dtype}")

    # Verify top-level re-export
    import cellstream
    assert hasattr(cellstream, 'phase_velocity'), "phase_velocity not re-exported at top level"
    assert not hasattr(cellstream, 'flow'), "flow module should not exist"
    print("\nTop-level re-exports OK.")

    print("\nAll tests passed successfully!")
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback; traceback.print_exc()
    exit(1)
