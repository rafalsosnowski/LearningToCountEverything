import numpy as np
import torch

class RegressorCalibrationHook:
    def __init__(self, save_path="calib_count_regressor.npy"):
        self.target_samples=1024
        self.save_path = save_path
        self.captured_tensors =[]
        self.is_done = False
        self.counter = 0

    def intercept(self, correlation_map):
        """
        Call this function inside your PyTorch loop right before the Regressor.
        Expected `correlation_map` shape from your original code: (Num_Exemplars, Channels, H, W)
        """
        if self.is_done:
            return

        # Move to CPU, convert to NumPy
        np_tensor = correlation_map.detach().cpu().numpy()
        
        # Reformat shape to match our ONNX Refactor: (Batch=1, Num_Exemplars, Channels, H, W)
        # NPU expects a static Batch dimension.

        # np_tensor = np.expand_dims(np_tensor, axis=0)
        # if self.counter == 0:
        np_tensor = np.squeeze(np_tensor, axis=0)
        self.captured_tensors.append(np_tensor)

        
        if len(self.captured_tensors) >= self.target_samples:
            self.save()
            self.is_done = True
        self.counter += 1

    def save(self):
        print(f"\n[Calibration Hook] Reached {self.target_samples} samples!")
        # Concatenate on Batch dimension -> (1000, Num_Exemplars, Channels, H, W)
        calib_dataset = np.concatenate(self.captured_tensors, axis=0)
        calib_dataset = calib_dataset[:self.target_samples]
        np.save(self.save_path, calib_dataset.astype(np.float32))
        print(f"[Calibration Hook] Saved {self.save_path}. Shape: {calib_dataset.shape}")
        raise Exception("Calibration complete!")