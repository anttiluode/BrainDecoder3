import os
import time
import logging
import hashlib
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import mne
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

# Additional scientific imports for frequency analysis and advanced computations
from scipy import signal
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is contiguous and has positive strides."""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.copy()

def check_tensor_device(tensor: torch.Tensor, expected_device: torch.device):
    """Check if the tensor is on the expected device."""
    if tensor.device != expected_device:
        raise RuntimeError(f"Tensor on wrong device: {tensor.device} vs {expected_device}")

def log_gpu_memory():
    """Log the current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"GPU Memory Allocated: {allocated:.2f} GB")
        logging.info(f"GPU Memory Reserved: {reserved:.2f} GB")

# -----------------------------------------------------------------------------
# Visualization State
# -----------------------------------------------------------------------------
@dataclass
class VisualizationState:
    """Holds the current state of visualization."""
    current_time: float = 0.0
    is_playing: bool = False
    seek_requested: bool = False
    seek_time: float = 0.0
    playback_speed: float = 1.0

# -----------------------------------------------------------------------------
# TextHandler for Logging
# -----------------------------------------------------------------------------
class TextHandler(logging.Handler):
    """This class allows you to log to a Tkinter Text widget."""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)

# -----------------------------------------------------------------------------
# Field Processor (Enhanced without Brain Constraints)
# -----------------------------------------------------------------------------
class FieldProcessor:
    """
    Processes EEG information using neural fields based on wave equations.
    Maps lower frequencies (theta/alpha) to coarse structure,
    and higher frequencies (beta/gamma) to fine details.
    """
    def __init__(self, resolution=512, field_dir='fields', dt=0.1,
                spatial_coupling=1.0, temporal_coupling=0.5, device='cpu'):
        self.resolution = resolution
        self.field_dir = field_dir
        os.makedirs(self.field_dir, exist_ok=True)
        self.device = device
        
        # Define frequency bands and their properties
        self.frequency_bands = {
            'theta': {  # 4-8 Hz - coarse structure
                'range': (4, 8),
                'scale_range': (0.5, 1.0),
                'transforms': 3,
                'detail_weight': 0.2,
                'original_scale_range': (0.5, 1.0)
            },
            'alpha': {  # 8-13 Hz - intermediate structure
                'range': (8, 13),
                'scale_range': (0.3, 0.6),
                'transforms': 4,
                'detail_weight': 0.4,
                'original_scale_range': (0.3, 0.6)
            },
            'beta': {  # 13-30 Hz - fine structure
                'range': (13, 30),
                'scale_range': (0.2, 0.4),
                'transforms': 5,
                'detail_weight': 0.6,
                'original_scale_range': (0.2, 0.4)
            },
            'gamma': {  # 30-100 Hz - finest details
                'range': (30, 100),
                'scale_range': (0.1, 0.3),
                'transforms': 6,
                'detail_weight': 0.8,
                'original_scale_range': (0.1, 0.3)
            }
        }
        
        # Field parameters
        self.field_params = {
            'dt': dt,
            'spatial_coupling': spatial_coupling,
            'temporal_coupling': temporal_coupling
        }
        
        # Field constants for the wave equation
        self.c = 1.0  # Wave speed
        self.alpha = 0.1  # Nonlinear damping
        self.beta = 0.05   # Coupling strength

    def generate_deterministic_seed(self, image_path: str, channel_name: str) -> int:
        """Generate a deterministic seed based on the image path and channel name."""
        combined = f"{image_path}_{channel_name}"
        hash_digest = hashlib.md5(combined.encode()).hexdigest()
        return int(hash_digest, 16) % (2**32)
    
    def analyze_eeg_frequencies(self, eeg_data: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract power in different frequency bands from 1D EEG data.

        :param eeg_data: 1D numpy array of the EEG signal (e.g., 1s window).
        :param fs: Sampling frequency of the EEG data.
        :return: dict of band_name -> average power
        """
        band_powers = {}
        # For each band, perform bandpass filtering and compute power
        for band_name, band_info in self.frequency_bands.items():
            low, high = band_info['range']
            nyq = fs / 2
            try:
                b, a = signal.butter(4, [low/nyq, high/nyq], btype='band', analog=False)
                # Filter signal
                filtered = signal.filtfilt(b, a, eeg_data)
                # Power calculation
                band_powers[band_name] = np.mean(filtered**2)
            except Exception as e:
                logging.error(f"Error in frequency analysis for band {band_name}: {e}")
                band_powers[band_name] = 0.0  # Assign zero power if error occurs
                
        return band_powers

    # In the FieldProcessor class, modify the generate_field_pattern method:

    def generate_field_pattern(self, band_powers: Dict[str, float], seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a combined neural field pattern where each band influences the field dynamics.
        
        :param band_powers: dict of band_name -> band power
        :param seed: Optional seed for deterministic field generation
        :return: field pattern as a NumPy array (float32, range ~ [0..1])
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        field = torch.zeros((self.resolution, self.resolution), device=self.device, dtype=torch.float32)
        
        total_power = sum(band_powers.values())
        if total_power < 1e-12:
            # If no power, or all-zero EEG, just return empty field
            return field.cpu().numpy()
        
        for band_name, power in band_powers.items():
            band_info = self.frequency_bands[band_name]
            # Relative power
            rel_power = power / total_power
            
            # 1) Compute field for this band
            band_field = self._compute_band_field(band_info, rel_power)
            
            # 2) Process field to add dynamics
            processed = self._process_band_field(band_field, band_info, rel_power)
            
            # 3) Weighted sum - ensure processed is on the same device as field
            processed_tensor = torch.tensor(processed, device=self.device, dtype=torch.float32)
            field += processed_tensor * band_info['detail_weight']
        
        # Normalize final result
        min_val = field.min()
        max_val = field.max()
        if (max_val - min_val) > 1e-12:
            field = (field - min_val) / (max_val - min_val)
        
        # Move to CPU before converting to numpy
        return field.cpu().numpy()

    def _compute_band_field(self, band_info: dict, relative_power: float) -> torch.Tensor:
        """
        Compute neural field for a single frequency band using the wave equation.
        
        :param band_info: dictionary containing band properties
        :param relative_power: relative power of the band
        :return: computed field tensor
        """
        # Initialize field u(x, y, t) and velocity v(x, y, t)
        u = torch.rand((self.resolution, self.resolution), device=self.device, dtype=torch.float32) * 0.1
        v = torch.zeros_like(u)
        
        # Time steps for simulation
        num_steps = 100  # Adjust as needed for dynamics
        
        # Grid spacing
        dx = 1.0
        dy = 1.0
        
        for step in range(num_steps):
            # Compute Laplacian using finite differences
            laplacian = (
                torch.roll(u, shifts=1, dims=0) +
                torch.roll(u, shifts=-1, dims=0) +
                torch.roll(u, shifts=1, dims=1) +
                torch.roll(u, shifts=-1, dims=1) -
                4 * u
            ) / (dx * dy)
            
            # Enhanced wave equation with phase coupling
            # ∂²u/∂t² = c²∇²u - αu³ - βv + ξ + Phase Coupling
            # Update velocity v
            quantum_noise = torch.randn_like(v, device=self.device) * 0.05
            # Compute phase using FFT
            fft_u = torch.fft.fft2(u)
            phase = torch.angle(fft_u)
            coupling = torch.real(torch.fft.ifft2(torch.exp(1j * phase))) * self.field_params['spatial_coupling']
            coupling = coupling.to(self.device)
            
            v_update = self.field_params['dt'] * (
                self.c**2 * laplacian - 
                self.alpha * (u ** 3) - 
                self.beta * v + 
                quantum_noise +
                coupling
            )
            v += v_update
            
            # Update field u
            u += self.field_params['dt'] * v
        
        # Apply relative power scaling
        u *= relative_power
        
        # Normalize
        u = (u - u.min()) / (u.max() - u.min() + 1e-12)
        return u

    def _process_band_field(self, field: torch.Tensor, band_info: dict, relative_power: float) -> np.ndarray:
        """
        Apply multi-scale processing to add details to the field.
        """
        # Create a Gaussian pyramid to extract details
        levels = 3
        pyramid = []
        current = field.cpu().numpy()

        for _ in range(levels):
            h, w = current.shape
            if min(h, w) < 2:
                break
            scaled = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            pyramid.append(scaled)
            current = scaled

        # Reconstruct with frequency-dependent weighting
        processed = np.zeros_like(field.cpu().numpy())
        size = (self.resolution, self.resolution)
        for i, level_img in enumerate(pyramid):
            # Higher-level pyramid = finer detail
            weight = relative_power * (1.0 - i / len(pyramid))
            up = cv2.resize(level_img, size, interpolation=cv2.INTER_LINEAR)
            processed += up * weight

        # Combine with the base field
        processed += field.cpu().numpy() * 0.5

        # Final normalize
        mx = processed.max()
        if mx > 1e-12:
            processed /= mx

        return processed

    def generate_and_save_field(self, eeg_data: np.ndarray, fs: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate neural field from EEG data and return it as a NumPy array.
        :param eeg_data: 1D numpy array of EEG data.
        :param fs: Sampling frequency.
        :param seed: Optional seed for deterministic field generation.
        :return: Field pattern as a NumPy array (float32, range ~ [0..1])
        """
        band_powers = self.analyze_eeg_frequencies(eeg_data, fs)
        logging.info(f"Band powers: {band_powers}")
        field_np = self.generate_field_pattern(band_powers, seed=seed)
        logging.info(f"Generated field pattern with shape: {field_np.shape}")
        return field_np
    
    def save_field_from_original(self, original_image_path: str, band_powers: Dict[str, float]) -> str:
        """
        Generate and save the field image corresponding to the original image.
        :param original_image_path: Path to the original image.
        :param band_powers: Band powers used for field generation.
        :return: Path to the saved field image.
        """
        field_tensor = torch.from_numpy(self.generate_field_pattern(band_powers)).float().to(self.device)
        field_np = field_tensor.cpu().numpy()
        field_uint8 = (field_np * 255).astype(np.uint8)
        field_pil = Image.fromarray(field_uint8)
        original_basename = os.path.basename(original_image_path)
        if original_basename.startswith('field_'):
            field_filename = original_basename  # Avoid double prefixing
        else:
            field_filename = f"field_{original_basename}"
        field_path = os.path.join(self.field_dir, field_filename)
        field_pil.save(field_path)
        logging.info(f"Saved field image: {field_path}")
        return field_path

# -----------------------------------------------------------------------------
# EEG Processing
# -----------------------------------------------------------------------------
class EEGProcessor:
    """Handles EEG data loading and retrieval."""
    def __init__(self, resolution: int = 512):
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        # 1 second window for retrieval
        self.window_size = 1.0  
        self.resolution = resolution

    def load_file(self, filepath: str) -> bool:
        """Load EEG data from an EDF file."""
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            logging.info(f"Loaded EEG file: {filepath}, Filenames: {self.raw.filenames}")
            return True
        except Exception as e:
            logging.error(f"Failed to load EEG file: {e}")
            return False

    def get_channels(self):
        """Return list of channel names."""
        if self.raw:
            return self.raw.ch_names
        return []

    def get_data(self, channel: int, start_time: float) -> Optional[np.ndarray]:
        """Retrieve 1s of EEG data for a specific channel and time."""
        if self.raw is None:
            return None
        try:
            start_sample = int(start_time * self.sfreq)
            samples_needed = int(self.window_size * self.sfreq)
            end_sample = start_sample + samples_needed
            if end_sample > self.raw.n_times:
                end_sample = self.raw.n_times
            data, _ = self.raw[channel, start_sample:end_sample]
            return data.flatten()
        except Exception as e:
            logging.error(f"Error getting EEG data: {e}")
            return None

# -----------------------------------------------------------------------------
# Enhanced U-Net Model (With Gradient Checkpointing and Mixed Precision Compatibility)
# -----------------------------------------------------------------------------
class EnhancedUNet(nn.Module):
    """U-Net for reversing field transformations (outputs grayscale)."""
    def __init__(self):
        super(EnhancedUNet, self).__init__()

        # Encoder
        self.enc1 = self._double_conv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)

        # Final conv => 1 channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """Helper for double convolution layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final_conv(d1)

# -----------------------------------------------------------------------------
# Dataset for Training
# -----------------------------------------------------------------------------
class ImagePairDataset(Dataset):
    """Dataset for image pairs (field and original)."""
    def __init__(self, original_dir, field_dir, image_pairs, transform=None):
        self.original_dir = original_dir
        self.field_dir = field_dir
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        field_file, orig_file = self.image_pairs[idx]  # Order: field first, then original

        # Load images
        field_img = Image.open(os.path.join(self.field_dir, field_file)).convert('L')
        original_img = Image.open(os.path.join(self.original_dir, orig_file)).convert('L')

        if self.transform:
            field_img = self.transform(field_img)
            original_img = self.transform(original_img)

        return field_img, original_img

# -----------------------------------------------------------------------------
# Decoder Testing (Integrated with Device Fix)
# -----------------------------------------------------------------------------
class DecoderTest:
    """Testing framework for field-to-image decoding."""
    def __init__(self, model, device, resolution=512, field_dir='fields'):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.field_dir = field_dir  # Directory where fields are stored
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        # We'll create one field processor for testing
        self.field_processor = FieldProcessor(resolution=self.resolution, field_dir=self.field_dir, device=self.device.type)

    def process_paired_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Test field encoding and decoding on a paired image (original and field).
        Returns: (original, field, decoded, PSNR, SSIM)
        """
        # Load and preprocess original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Invalid image or cannot open: {image_path}")

        # Resize to match the resolution
        original = cv2.resize(original, (self.resolution, self.resolution))

        # Analyze EEG frequencies (simulated here as uniform band powers for demonstration)
        # In actual use, band_powers should come from EEG data
        band_powers = {
            'theta': 1.0,
            'alpha': 0.5,
            'beta': 1.5,
            'gamma': 2.0
        }

        # Generate field pattern using in-memory processing
        field_np = self.field_processor.generate_field_pattern(band_powers)
        logging.info(f"Generated field pattern for paired image with shape: {field_np.shape}")

        # Decode field
        field_tensor = torch.from_numpy(field_np).float().to(self.device)
        field_tensor = field_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Ensure tensor is contiguous
        field_tensor = field_tensor.contiguous()

        # Error checking
        check_tensor_device(field_tensor, self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                decoded_tensor = self.model(field_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                if decoded_np.shape[0] > 1:
                    logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
                decoded_np = decoded_np[0]  # channel 0
                decoded_np = (decoded_np * 255).clip(0, 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        # Metrics
        if original.shape != decoded_np.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape}, decoded={decoded_np.shape}. "
                "They must be identical for PSNR/SSIM."
            )

        psnr_val = psnr(original, decoded_np)
        ssim_val = ssim(original, decoded_np)

        return original, field_np, decoded_np, psnr_val, ssim_val

    def decode_any_field_image(self, field_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode any standalone field image without needing a paired original.
        Returns: (field, decoded)
        """
        field = cv2.imread(field_path, cv2.IMREAD_GRAYSCALE)
        if field is None:
            raise ValueError(f"Invalid field image or cannot open: {field_path}")

        # Resize field to match resolution if necessary
        field = cv2.resize(field, (self.resolution, self.resolution))

        # Decode field
        field_tensor = self.transform(Image.fromarray(field)).unsqueeze(0).to(self.device)

        # Ensure tensor is contiguous
        field_tensor = field_tensor.contiguous()

        # Error checking
        check_tensor_device(field_tensor, self.device)
        log_gpu_memory()

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                decoded_tensor = self.model(field_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                if decoded_np.shape[0] > 1:
                    logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
                decoded_np = decoded_np[0]  # channel 0
                decoded_np = (decoded_np * 255).clip(0, 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        return field, decoded_np

# -----------------------------------------------------------------------------
# Video Recording
# -----------------------------------------------------------------------------
class VideoRecorder:
    """Handles recording of EEG visualization to video."""
    def __init__(self, resolution=512):
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.output_path = None

    def start_recording(self, output_path: str, fps: float = 30.0):
        """Start recording video."""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.resolution * 2, self.resolution)
        )
        self.is_recording = True
        logging.info(f"Started recording to {output_path}")

    def add_frame(self, field_frame: np.ndarray, decoded_frame: np.ndarray):
        """Add a frame to the video (side-by-side)."""
        if not self.is_recording:
            return

        field_frame = cv2.resize(field_frame, (self.resolution, self.resolution))
        decoded_frame = cv2.resize(decoded_frame, (self.resolution, self.resolution))
        combined = np.hstack([field_frame, decoded_frame])
        self.writer.write(combined)

    def stop_recording(self):
        """Stop recording and save video."""
        if self.writer:
            self.writer.release()
        self.is_recording = False
        self.writer = None
        logging.info(f"Stopped recording and saved to {self.output_path}")

# -----------------------------------------------------------------------------
# Results Logger
# -----------------------------------------------------------------------------
class ResultsLogger:
    """Logs and tracks decoder performance metrics."""
    def __init__(self, log_dir: str = "decoder_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = []

    def log_result_paired(self, original_path: str, psnr_val: float, ssim_val: float):
        """Log metrics for a paired decode attempt."""
        self.metrics.append({
            'type': 'Paired',
            'image': original_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'timestamp': time.time()
        })
        logging.info(f"Logged paired results for {original_path}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

    def log_result_standalone(self, field_path: str, decoded_image: np.ndarray):
        """Log metrics for a standalone decode attempt."""
        # Since there's no original image, metrics like PSNR and SSIM are not applicable
        self.metrics.append({
            'type': 'Standalone',
            'image': field_path,
            'decoded': decoded_image,
            'timestamp': time.time()
        })
        logging.info(f"Logged standalone decode for {field_path}")

    def save_metrics(self):
        """Save all metrics to file."""
        if not self.metrics:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(self.log_dir, f"metrics_{timestamp}.txt")
        with open(log_path, 'w') as f:
            for m in self.metrics:
                if m['type'] == 'Paired':
                    f.write(f"{m['type']},{m['image']},{m['psnr']:.4f},{m['ssim']:.4f}\n")
                elif m['type'] == 'Standalone':
                    f.write(f"{m['type']},{m['image']},Decoded Image Saved\n")
        logging.info(f"Saved metrics to {log_path}")

    def get_summary(self) -> dict:
        """Get summary statistics of all paired metrics."""
        psnr_vals = [m['psnr'] for m in self.metrics if m['type'] == 'Paired']
        ssim_vals = [m['ssim'] for m in self.metrics if m['type'] == 'Paired']

        if not psnr_vals:
            return {'psnr_avg': 0, 'ssim_avg': 0, 'psnr_std': 0, 'ssim_std': 0}

        return {
            'psnr_avg': np.mean(psnr_vals),
            'psnr_std': np.std(psnr_vals),
            'ssim_avg': np.mean(ssim_vals),
            'ssim_std': np.std(ssim_vals)
        }

# -----------------------------------------------------------------------------
# EEGNoids Application Class (Enhanced without Biological Bridge)
# -----------------------------------------------------------------------------
class EEGNoidsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEGNoids - EEG Neural Organoid Visualizer")
        
        # Initialize components
        self.resolution = 512
        self.eeg = EEGProcessor(resolution=self.resolution)
        self.state = VisualizationState()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Explicitly use cuda:0
        self.model = EnhancedUNet().to(self.device)
        self.model.eval()  # default to eval mode
        
        self.last_update = time.time()
        
        # Initialize field processor with a specified field directory
        self.field_dir = 'fields'
        self.field_processor = FieldProcessor(
            resolution=self.resolution,
            field_dir=self.field_dir,
            dt=0.1,
            spatial_coupling=1.0,
            temporal_coupling=0.5,
            device=self.device.type
        )
        
        # Initialize additional components
        self.decoder_test = DecoderTest(self.model, self.device, self.resolution, field_dir=self.field_dir)
        self.video_recorder = VideoRecorder(self.resolution)
        self.results_logger = ResultsLogger()
        
        self.setup_gui()

    def setup_gui(self):
        """Set up the graphical user interface."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG", command=self.load_eeg)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Testing menu
        test_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Testing", menu=test_menu)
        test_menu.add_command(label="Test Paired Image", command=self.test_paired_image)
        test_menu.add_command(label="Decode Field Image", command=self.decode_field_image)
        test_menu.add_command(label="Test Batch Images", command=self.test_batch_images)
        test_menu.add_command(label="View Test Results", command=self.view_test_results)

        # Processing menu
        process_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=process_menu)
        process_menu.add_command(label="Batch Process Images", command=self.batch_process)
        process_menu.add_command(label="Train Model", command=self.train_model)

        # Recording menu
        record_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Recording", menu=record_menu)
        record_menu.add_command(label="Start Recording", command=self.start_recording)
        record_menu.add_command(label="Stop Recording", command=self.stop_recording)

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel
        control_frame = ttk.LabelFrame(main_container, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(control_frame, text="EEG Channel:").pack(pady=5)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var)
        self.channel_combo.pack(fill=tk.X, padx=5, pady=5)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_selected)  # Bind event

        play_frame = ttk.Frame(control_frame)
        play_frame.pack(fill=tk.X, pady=5)
        self.play_btn = ttk.Button(play_frame, text="Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Time Position (s):").pack(pady=5)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(control_frame, from_=0, to=100,
                                     variable=self.time_var, command=self.seek)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)

        # Playback Speed Control
        ttk.Label(control_frame, text="Playback Speed:").pack(pady=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            control_frame, from_=0.5, to=2.0,
            variable=self.speed_var,
            command=self.update_playback_speed
        )
        speed_slider.pack(fill=tk.X, padx=5, pady=5)
        speed_slider.set(1.0)  # Reset to default

        # Frequency Controls
        self.add_frequency_controls(control_frame)

        # Right visualization panel
        viz_frame = ttk.LabelFrame(main_container, text="Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        field_frame = ttk.Frame(viz_frame)
        field_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(field_frame, text="Neural Field Pattern").pack()
        self.field_canvas = tk.Canvas(field_frame, bg='black', width=512, height=512)
        self.field_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        decoded_frame = ttk.Frame(viz_frame)
        decoded_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(decoded_frame, text="Decoded Image").pack()
        self.decoded_canvas = tk.Canvas(decoded_frame, bg='black', width=512, height=512)
        self.decoded_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add Logging Widget
        self.add_logging_widget()

    def add_frequency_controls(self, ctrl_frame):
        """Add frequency band control sliders to GUI"""
        freq_frame = ttk.LabelFrame(ctrl_frame, text="Frequency Controls")
        freq_frame.pack(fill=tk.X, padx=5, pady=5)

        self.freq_controls = {}

        # For each frequency band, add power, coupling, and frequency sliders
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            band_frame = ttk.LabelFrame(freq_frame, text=f"{band.title()}")
            band_frame.pack(fill=tk.X, padx=5, pady=2)

            # Power control
            ttk.Label(band_frame, text="Power:").pack(anchor='w')
            power_var = tk.DoubleVar(value=1.0)
            power_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=power_var,
                command=lambda v, b=band: self.update_band_power(b, float(v))
            )
            power_slider.pack(fill=tk.X, padx=5, pady=2)
            power_slider.set(1.0)  # Reset to default

            # Coupling control
            ttk.Label(band_frame, text="Coupling:").pack(anchor='w')
            coupling_var = tk.DoubleVar(value=1.0)
            coupling_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=coupling_var,
                command=lambda v, b=band: self.update_band_coupling(b, float(v))
            )
            coupling_slider.pack(fill=tk.X, padx=5, pady=2)
            coupling_slider.set(1.0)  # Reset to default

            # Oscillation frequency control
            ttk.Label(band_frame, text="Frequency (Hz):").pack(anchor='w')
            freq_var = tk.DoubleVar(value=(
                self.field_processor.frequency_bands[band]['range'][0] + 
                self.field_processor.frequency_bands[band]['range'][1]
            ) / 2)
            freq_slider = ttk.Scale(
                band_frame, 
                from_=self.field_processor.frequency_bands[band]['range'][0],
                to=self.field_processor.frequency_bands[band]['range'][1],
                variable=freq_var,
                command=lambda v, b=band: self.update_band_frequency(b, float(v))
            )
            freq_slider.pack(fill=tk.X, padx=5, pady=2)
            freq_slider.set(freq_var.get())  # Reset to default

            self.freq_controls[band] = {
                'power': power_var,
                'coupling': coupling_var,
                'frequency': freq_var
            }

    def update_playback_speed(self, value):
        """Update playback speed based on slider."""
        self.state.playback_speed = float(value)
        logging.info(f"Playback speed set to {self.state.playback_speed}")

    def update_band_power(self, band, value):
        """Update band power scaling"""
        # Reset to original scale ranges
        original_scale_min, original_scale_max = self.field_processor.frequency_bands[band]['original_scale_range']
        # Apply scaling based on slider value
        new_scale_min = original_scale_min * value
        new_scale_max = original_scale_max * value
        self.field_processor.frequency_bands[band]['scale_range'] = (new_scale_min, new_scale_max)
        logging.info(f"Updated power scaling for {band} band to {value}")
        self.update_visualization()

    def update_band_coupling(self, band, value):
        """Update coupling strength for band"""
        # Here, coupling influences the spatial_coupling parameter
        self.field_processor.field_params['spatial_coupling'] = value
        logging.info(f"Updated spatial coupling for {band} band to {value}")
        self.update_visualization()

    def update_band_frequency(self, band, value):
        """Update oscillation frequency"""
        # Update the frequency range based on the center frequency
        band_info = self.field_processor.frequency_bands[band]
        center_freq = value
        bandwidth = (band_info['range'][1] - band_info['range'][0]) / 2
        new_low = max(center_freq - bandwidth / 2, 0.1)  # Prevent negative frequencies
        new_high = center_freq + bandwidth / 2
        if new_high <= new_low:
            new_high = new_low + 0.1  # Ensure high > low

        # Update the frequency range
        band_info['range'] = (new_low, new_high)
        logging.info(f"Updated frequency range for {band} band to {new_low:.2f} - {new_high:.2f} Hz")
        self.update_visualization()

    def on_channel_selected(self, event):
        """Handle channel selection change."""
        selected_channel = self.channel_var.get()
        logging.info(f"Selected channel: {selected_channel}")
        
        # Reset current time and seek state
        self.state.current_time = 0.0
        self.state.seek_requested = False
        self.state.seek_time = 0.0
        
        if not self.state.is_playing:
            # If paused, reset the visualizations to black
            black_field = Image.new('RGB', (512, 512), color='black')
            black_field_photo = ImageTk.PhotoImage(black_field)
            self.field_canvas.delete("all")
            self.field_canvas.create_image(0, 0, image=black_field_photo, anchor=tk.NW)
            self.field_canvas.image = black_field_photo  # Keep reference
            
            black_decoded = Image.new('RGB', (512, 512), color='black')
            black_decoded_photo = ImageTk.PhotoImage(black_decoded)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=black_decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = black_decoded_photo  # Keep reference
        else:
            # If playing, reset the current visualization
            self.update_visualization()

    def toggle_playback(self):
        """Toggle between play and pause."""
        self.state.is_playing = not self.state.is_playing
        self.play_btn.configure(text="Pause" if self.state.is_playing else "Play")
        if self.state.is_playing:
            self.last_update = time.time()
            self.update()

    def seek(self, value):
        """Handle user seeking to a different time."""
        if not self.state.is_playing:
            self.state.seek_requested = True
            self.state.seek_time = float(value)
            self.state.current_time = float(value)
            self.update_visualization()

    def update(self):
        """Update loop called periodically during playback."""
        if not self.state.is_playing:
            return

        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        self.state.current_time += dt * self.state.playback_speed

        # Handle end of EEG data
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0  # Loop back to start

        # Update the time slider
        self.time_var.set(self.state.current_time)

        # Update visualization
        self.update_visualization()

        # Schedule next update (~30 FPS)
        self.root.after(33, self.update)

    # Modify the update_visualization method in EEGNoidsApp class:
    def update_visualization(self):
        """Compute field from current EEG snippet and decode it."""
        if not self.eeg.raw or not self.channel_var.get():
            return
        try:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
            data = self.eeg.get_data(channel_idx, self.state.current_time)
            if data is None:
                return

            # Generate deterministic seed
            seed = self.field_processor.generate_deterministic_seed(
                self.eeg.raw.filenames[0] if hasattr(self.eeg.raw, 'filenames') and len(self.eeg.raw.filenames) > 0 else "default_seed",
                self.channel_var.get()
            )

            # Get band powers from EEG data
            band_powers = self.field_processor.analyze_eeg_frequencies(data, self.eeg.sfreq)
            
            # Generate field pattern - ensure it returns a tensor
            field_tensor = torch.tensor(self.field_processor.generate_field_pattern(band_powers, seed=seed), 
                                    device=self.device, dtype=torch.float32)
            
            # Create colored visualization of the field
            with torch.no_grad():
                field_np = field_tensor.cpu().numpy()
                colored_field = cv2.applyColorMap((field_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                field_img = Image.fromarray(cv2.cvtColor(colored_field, cv2.COLOR_BGR2RGB))
                field_img = field_img.resize((512, 512), Image.LANCZOS)
            
            # Update field canvas
            self.current_field_photo = ImageTk.PhotoImage(field_img)
            self.field_canvas.delete("all")
            self.field_canvas.create_image(0, 0, image=self.current_field_photo, anchor=tk.NW)
            self.field_canvas.update()

            # Prepare tensor for decoding
            decode_tensor = field_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Decode field
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                with torch.no_grad():
                    decoded = self.model(decode_tensor)
                    decoded = decoded.squeeze()
                    # Keep tensor operations on GPU as long as possible
                    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
                    # Only move to CPU for final visualization
                    decoded_np = (decoded.cpu().numpy() * 255).astype(np.uint8)
                    decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

            # Update decoded image display
            decoded_img = Image.fromarray(decoded_np)
            decoded_img = decoded_img.resize((512, 512), Image.LANCZOS)
            
            self.current_decoded_photo = ImageTk.PhotoImage(decoded_img)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=self.current_decoded_photo, anchor=tk.NW)
            self.decoded_canvas.update()

            # Handle recording if active
            if self.video_recorder.is_recording:
                decoded_bgr = cv2.cvtColor(decoded_np, cv2.COLOR_GRAY2BGR)
                self.video_recorder.add_frame(colored_field, decoded_bgr)

        except Exception as e:
            logging.error(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_logging_widget(self):
        """Add a text widget to display log messages."""
        log_frame = ttk.LabelFrame(self.root, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        # Add a scrolled text widget
        self.log_text = tk.Text(log_frame, height=10, state='disabled', wrap='word')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scrollbar.set

        # Redirect logging to the text widget
        logging.getLogger().addHandler(TextHandler(self.log_text))

    # -------------------------------------------------------------------------
    # GUI Command Methods
    # -------------------------------------------------------------------------
    def load_eeg(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if filepath and self.eeg.load_file(filepath):
            channels = self.eeg.get_channels()
            if not channels:
                messagebox.showerror("Error", "No channels found in EEG file.")
                return
            self.channel_combo['values'] = channels
            self.channel_combo.set(channels[0])
            self.time_slider.configure(to=self.eeg.duration)
            messagebox.showinfo("Success", "EEG file loaded successfully")
            logging.info(f"EEG file loaded: {filepath}")
            self.update_visualization()
        else:
            messagebox.showerror("Error", "Failed to load EEG file")
            logging.error("Failed to load EEG file")

    def test_paired_image(self):
        """Test decoding on a paired original-field image."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            results_window = tk.Toplevel(self.root)
            results_window.title("Decoder Test Results (Paired)")

            original, field_np, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(filepath)

            images_frame = ttk.Frame(results_window)
            images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Original Image
            original_frame = ttk.LabelFrame(images_frame, text="Original Image")
            original_frame.pack(side=tk.LEFT, padx=5)
            original_canvas = tk.Canvas(original_frame, width=256, height=256)
            original_canvas.pack()
            original_pil = Image.fromarray(original)
            original_pil = original_pil.resize((256, 256), Image.LANCZOS)
            original_photo = ImageTk.PhotoImage(original_pil)
            original_canvas.create_image(0, 0, image=original_photo, anchor=tk.NW)
            original_canvas.image = original_photo

            # Field Image
            field_frame = ttk.LabelFrame(images_frame, text="Neural Field")
            field_frame.pack(side=tk.LEFT, padx=5)
            field_canvas = tk.Canvas(field_frame, width=256, height=256)
            field_canvas.pack()
            colored_field = cv2.applyColorMap((field_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            field_pil = Image.fromarray(cv2.cvtColor(colored_field, cv2.COLOR_BGR2RGB))
            field_pil = field_pil.resize((256, 256), Image.LANCZOS)
            field_photo = ImageTk.PhotoImage(field_pil)
            field_canvas.create_image(0, 0, image=field_photo, anchor=tk.NW)
            field_canvas.image = field_photo

            # Decoded Image
            decoded_frame = ttk.LabelFrame(images_frame, text="Decoded Image")
            decoded_frame.pack(side=tk.LEFT, padx=5)
            decoded_canvas = tk.Canvas(decoded_frame, width=256, height=256)
            decoded_canvas.pack()
            decoded_pil = Image.fromarray(decoded)
            decoded_pil = decoded_pil.resize((256, 256), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(decoded_pil)
            decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            decoded_canvas.image = decoded_photo

            # Metrics
            metrics_frame = ttk.Frame(results_window)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(metrics_frame, text=f"PSNR: {psnr_val:.2f} dB").pack(side=tk.LEFT, padx=10)
            ttk.Label(metrics_frame, text=f"SSIM: {ssim_val:.4f}").pack(side=tk.LEFT, padx=10)

            # Log results
            self.results_logger.log_result_paired(filepath, psnr_val, ssim_val)

        except Exception as e:
            logging.error(f"Error in paired decoder test: {e}")
            messagebox.showerror("Error", f"Paired Test failed: {str(e)}")

    def decode_field_image(self):
        """Decode any standalone field image without needing a paired original."""
        field_path = filedialog.askopenfilename(
            title="Select Field Image to Decode",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not field_path:
            return
        try:
            # Decode field
            field, decoded = self.decoder_test.decode_any_field_image(field_path)

            # Show field in the left canvas
            colored_field = cv2.applyColorMap(field, cv2.COLORMAP_JET)
            field_pil = Image.fromarray(cv2.cvtColor(colored_field, cv2.COLOR_BGR2RGB))
            field_pil = field_pil.resize((512, 512), Image.LANCZOS)
            field_photo = ImageTk.PhotoImage(image=field_pil)
            self.field_canvas.delete("all")
            self.field_canvas.create_image(0, 0, image=field_photo, anchor=tk.NW)
            self.field_canvas.image = field_photo  # Keep reference

            # Show decoded image in the right canvas
            decoded_pil = Image.fromarray(decoded)
            decoded_pil = decoded_pil.resize((512, 512), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(image=decoded_pil)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = decoded_photo  # Keep reference

            # Log standalone decode
            self.results_logger.log_result_standalone(field_path, decoded)

            # If recording, add frame
            if self.video_recorder.is_recording:
                self.video_recorder.add_frame(colored_field, decoded)

            # Inform user
            messagebox.showinfo("Success", f"Decoded field image:\n{field_path}")

        except Exception as e:
            logging.error(f"Error in decoding field image: {e}")
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")

    def test_batch_images(self):
        """Test decoding on a batch of paired images."""
        input_dir = filedialog.askdirectory(title="Select Directory with Test Images")
        if not input_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Testing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Testing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                path_ = os.path.join(input_dir, filename)
                try:
                    original, field_np, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(path_)
                    self.results_logger.log_result_paired(path_, psnr_val, ssim_val)
                except Exception as e:
                    logging.error(f"Error processing {path_}: {e}")
                    continue

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            self.results_logger.save_metrics()
            progress_window.destroy()
            messagebox.showinfo("Success", "Batch testing complete!")

        except Exception as e:
            logging.error(f"Error in batch testing: {e}")
            messagebox.showerror("Error", f"Batch testing failed: {str(e)}")

    def view_test_results(self):
        """View summary of test results."""
        summary = self.results_logger.get_summary()
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Test Results Summary")

        ttk.Label(summary_window, text=f"Average PSNR: {summary['psnr_avg']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"PSNR Std Dev: {summary['psnr_std']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"Average SSIM: {summary['ssim_avg']:.4f}").pack(pady=5)
        ttk.Label(summary_window, text=f"SSIM Std Dev: {summary['ssim_std']:.4f}").pack(pady=5)

        results_frame = ttk.Frame(summary_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        results_listbox = tk.Listbox(results_frame, yscrollcommand=scrollbar.set, width=100)
        for metric in self.results_logger.metrics:
            if metric['type'] == 'Paired':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | {os.path.basename(metric['image'])} | "
                    f"PSNR: {metric['psnr']:.2f} dB | SSIM: {metric['ssim']:.4f}"
                )
            elif metric['type'] == 'Standalone':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | Standalone Decode | {os.path.basename(metric['image'])} | Decoded Image"
                )
        results_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=results_listbox.yview)

    def batch_process(self):
        """Create a batch of neural field patterns from images for training."""
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in input directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Batch Processing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Processing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = image.astype(float) / 255.0

                # For demonstration: compute band powers (assuming uniform band powers)
                band_powers = {
                    'theta': 1.0,
                    'alpha': 0.5,
                    'beta': 1.5,
                    'gamma': 2.0
                }
                # Generate and save field
                field_np = self.field_processor.generate_field_pattern(band_powers)
                logging.info(f"Generated field pattern for {filename} with shape: {field_np.shape}")

                # Create and save colored field image
                colored_field = cv2.applyColorMap((field_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                output_path = os.path.join(output_dir, f"field_{filename}")
                cv2.imwrite(output_path, colored_field)
                logging.info(f"Saved colored field image to {output_path}")

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", "Batch processing complete!")

        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
            
    def train_model(self):
        """Train the U-Net model directly on field/image pairs, bypassing bioprocessor."""
        original_dir = filedialog.askdirectory(title="Select Original Images Directory")
        if not original_dir:
            return
        field_dir = filedialog.askdirectory(title="Select Field Images Directory")
        if not field_dir:
            return

        try:
            # Find matching pairs of images
            original_files = [
                f for f in os.listdir(original_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            field_files = [f"field_{f}" for f in original_files]

            # Validate pairs exist
            valid_pairs = []
            for orig, field in zip(original_files, field_files):
                if os.path.exists(os.path.join(field_dir, field)):
                    valid_pairs.append((field, orig))

            if not valid_pairs:
                messagebox.showerror("Error", "No matching image pairs found")
                return

            # Setup progress tracking
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training Progress")
            progress_window.geometry("400x200")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Preparing training...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            # Training parameters
            num_epochs = 1200
            batch_size = 8  # Increased since we have more memory available
            learning_rate = 0.001

            # Data preprocessing
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            # Split data
            split_idx = int(0.8 * len(valid_pairs))
            train_pairs = valid_pairs[:split_idx]
            val_pairs = valid_pairs[split_idx:]

            # Create datasets
            train_dataset = ImagePairDataset(original_dir, field_dir, train_pairs, transform)
            val_dataset = ImagePairDataset(original_dir, field_dir, val_pairs, transform)

            # Create dataloaders with num_workers for faster loading
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )

            # Training setup
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            
            # Initialize gradient scaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

            for epoch in range(num_epochs):
                train_loss = 0.0
                self.model.train()

                # Training loop
                for batch_idx, (field_imgs, original_imgs) in enumerate(train_loader):
                    # Move data to device efficiently
                    field_imgs = field_imgs.to(self.device, non_blocking=True)
                    original_imgs = original_imgs.to(self.device, non_blocking=True)

                    # Forward pass with mixed precision
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        outputs = self.model(field_imgs)
                        loss = criterion(outputs, original_imgs)

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                    # Update progress
                    batch_progress = 100 * (batch_idx + 1) / len(train_loader)
                    progress_var.set((epoch + batch_progress/100) / num_epochs * 100)
                    status_var.set(f"Epoch {epoch+1}/{num_epochs} - Training...")
                    progress_window.update()

                # Validation loop
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for field_imgs, original_imgs in val_loader:
                        field_imgs = field_imgs.to(self.device, non_blocking=True)
                        original_imgs = original_imgs.to(self.device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                            outputs = self.model(field_imgs)
                            loss = criterion(outputs, original_imgs)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, 'best_model.pth')
                    logging.info(f"Saved best model with Val Loss: {val_loss:.4f}")

                # Save periodic checkpoint every 50 epochs
                if (epoch + 1) % 50 == 0:
                    checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss / len(train_loader),
                        'best_val_loss': best_val_loss
                    }, checkpoint_path)
                    logging.info(f"Saved periodic checkpoint at epoch {epoch+1}")

                # Save latest model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss / len(train_loader),
                    'best_val_loss': best_val_loss
                }, 'latest_model.pth')

                status_var.set(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", f"Training complete!\nBest validation loss: {best_val_loss:.4f}")

        except Exception as e:
            logging.error(f"Error in training: {e}")
            messagebox.showerror("Error", f"Training processing failed: {str(e)}")

    # -----------------------------------------------------------------------------
    # Recording Methods
    # -----------------------------------------------------------------------------
    def start_recording(self):
        """Start recording the visualization."""
        if self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Already recording!")
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not output_path:
            return
        try:
            self.video_recorder.start_recording(output_path)
            messagebox.showinfo("Recording", f"Recording started: {output_path}")
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        """Stop recording the visualization."""
        if not self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Not currently recording!")
            return
        try:
            self.video_recorder.stop_recording()
            messagebox.showinfo("Recording", "Recording stopped and saved.")
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")

    def load_model(self):
        """Load a trained model from a file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                messagebox.showinfo("Success", "Model loaded successfully!")
                logging.info(f"Model loaded from {filepath}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def save_model(self):
        """Save the current model to a file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                torch.save({'model_state_dict': self.model.state_dict()}, filepath)
                messagebox.showinfo("Success", "Model saved successfully!")
                logging.info(f"Model saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    # Configure logging
    logging.basicConfig(
        filename='eegnoids_app.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    root = tk.Tk()
    app = EEGNoidsApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()
