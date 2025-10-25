import torch
import numpy as np
import pydicom
from pathlib import Path
import cv2
from io import BytesIO
from PIL import Image
import joblib
import json
import re
from train_unet_pancreas import UNet  # reuse your UNet class

def load_unet_model(model_path, device=None, in_channels=1, base_ch=32):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(in_channels=in_channels, base_ch=base_ch)
    ckpt = torch.load(str(model_path), map_location=device)
    
    # Handle both model state_dict or raw state_dict
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    unet.load_state_dict(state)
    unet.to(device).eval()
    return unet, device

def read_dcm_to_image(path_or_folder):
    p = Path(path_or_folder)
    if p.is_dir():
        # Support both single DICOM files or folders
        files = sorted([f for f in p.glob("*") if f.suffix.lower() in [".dcm"]])
        if not files:
            raise FileNotFoundError("No DICOM files found in folder.")
        # Pick middle slice
        sel = files[len(files)//2]
        ds = pydicom.dcmread(str(sel))
    else:
        if p.suffix.lower() == '.dcm':
            ds = pydicom.dcmread(str(p))
        else:
            # Choose middle slice if multiple files
            if p.suffix.lower() == '.npy':
                arr = np.load(str(p))
                return arr.astype(np.float32)
            else:
                # Try to load as image
                arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise ValueError(f"Cannot load file: {p}")
                if len(arr.shape) == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                return arr.astype(np.float32)
    
    arr = ds.pixel_array.astype(np.float32)
    
    # Basic DICOM preprocessing
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    
    return arr

def preprocess_img(img, out_size=256):
    """
    Letterbox resize with metadata tracking (like crop.py)
    """
    h, w = img.shape[:2]
    scale = min(out_size / h, out_size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # Resize
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    top = (out_size - nh) // 2
    bottom = out_size - nh - top
    left = (out_size - nw) // 2
    right = out_size - nw - left
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=0)
    
    # Normalize to [0, 1]
    if img_padded.max() > 1:
        img_padded = img_padded.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(img_padded).unsqueeze(0).float()
    
    # Metadata for reconstruction
    letterbox_meta = {
        'scale': scale,
        'pad': [top, bottom, left, right],
        'resized_shape': [nh, nw],
        'final_shape': [out_size, out_size]
    }
    
    return tensor, letterbox_meta

def reconstruct_mask_from_crop(mask_pred, metadata):
    """
    Reconstruct mask from cropped prediction back to original image space
    
    Args:
        mask_pred: Predicted mask (256, 256) from model
        metadata: Combined metadata from crop+letterbox operations
        
    Returns:
        mask_full: Mask in original image coordinate system
    """
    crop_meta = metadata['crop_meta']
    letterbox_meta = metadata['letterbox_meta']
    
    # Step 1: Remove letterbox padding
    pad = letterbox_meta['pad']  # [top, bottom, left, right]
    top, bottom, left, right = pad
    
    # Remove padding from prediction
    h_final, w_final = letterbox_meta['final_shape']  # [256, 256]
    h_content = h_final - top - bottom
    w_content = w_final - left - right
    
    mask_unpadded = mask_pred[top:top+h_content, left:left+w_content]
    
    # Step 2: Resize back to crop size
    crop_shape = crop_meta['crop_shape']  # [H_crop, W_crop]
    mask_crop_size = cv2.resize(mask_unpadded, (crop_shape[1], crop_shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Step 3: Place back to original image coordinates
    if not crop_meta['has_foreground']:
        # Was full image, just resize to original
        orig_shape = crop_meta['original_shape']
        mask_full = cv2.resize(mask_crop_size, (orig_shape[1], orig_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    else:
        # Place crop back to original position
        orig_shape = crop_meta['original_shape']  # [H, W]
        bbox = crop_meta['crop_bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        # Create full canvas
        mask_full = np.zeros(orig_shape, dtype=mask_crop_size.dtype)
        mask_full[y1:y2, x1:x2] = mask_crop_size
    
    return mask_full

def find_matching_metadata(patient_id, slice_index, cropped_dir="./cropped"):
    """
    Find matching metadata file for a given patient/slice
    """
    cropped_path = Path(cropped_dir)
    
    # Search in both train and val metadata folders
    meta_dirs = [
        cropped_path / "train" / "metadata",
        cropped_path / "val" / "metadata"
    ]
    
    for meta_dir in meta_dirs:
        if not meta_dir.exists():
            continue
            
        meta_files = list(meta_dir.glob("*.json"))
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                # Match patient_id and slice_index
                if (str(metadata['patient_id']) == str(patient_id) and 
                    int(metadata['slice_index']) == int(slice_index)):
                    return metadata
            except Exception as e:
                continue
    
    return None

def extract_patient_slice_from_filename(filepath):
    """
    Extract patient_id and slice_index from various filename formats
    """
    filename = Path(filepath).stem
    
    # Try different patterns
    patterns = [
        r'PANCREAS_(\d+).*?(\d+)',  # PANCREAS_001_slice_050
        r'(\d+).*?(\d+)',           # Generic numbers
        r'[pP]atient.*?(\d+).*?[sS]lice.*?(\d+)',  # patient_001_slice_050
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            patient_num = int(match.group(1))
            slice_num = int(match.group(2))
            patient_id = f"PANCREAS_{patient_num:04d}"
            return patient_id, slice_num
    
    return None, None

def estimate_pancreas_crop(img, margin=50):
    """
    Estimate pancreas region when no metadata available
    Use center region as rough estimate
    """
    h, w = img.shape[:2]
    
    # Rough pancreas location (center-left region)
    x1 = max(0, w//4 - margin)
    y1 = max(0, h//3 - margin) 
    x2 = min(w, 3*w//4 + margin)
    y2 = min(h, 2*h//3 + margin)
    
    crop_meta = {
        'original_shape': [h, w],
        'crop_bbox': [x1, y1, x2, y2],
        'crop_shape': [y2-y1, x2-x1],
        'has_foreground': True
    }
    
    return crop_meta

def postprocess_pred(pred_mask, threshold=0.65):
    """
    Enhanced post-processing with morphological operations
    """
    # Threshold
    binary_mask = pred_mask >= threshold
    
    try:
        from skimage import morphology, measure
        
        # Remove small components
        labeled_mask = measure.label(binary_mask)
        props = measure.regionprops(labeled_mask)
        
        filtered_mask = np.zeros_like(binary_mask)
        for prop in props:
            # Keep only reasonable pancreas sizes (adjust based on your data)
            if 100 <= prop.area <= 15000 and prop.solidity >= 0.5:
                filtered_mask[labeled_mask == prop.label] = True
        
        # Morphological cleanup
        if np.any(filtered_mask):
            kernel = morphology.disk(2)
            filtered_mask = morphology.binary_opening(filtered_mask, kernel)
            filtered_mask = morphology.binary_closing(filtered_mask, kernel)
            binary_mask = filtered_mask
            
    except ImportError:
        print("scikit-image not available, using basic thresholding")
    
    mask_bin = binary_mask.astype(np.uint8) * 255
    return mask_bin

def overlay_mask_on_image(img, mask, alpha=0.4):
    """
    Overlay mask on image
    """
    if img.max() <= 1.0:
        img_display = (img * 255).astype(np.uint8)
    else:
        img_display = img.astype(np.uint8)
    
    # Convert to RGB
    if len(img_display.shape) == 2:
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_display.copy()
    
    # Create colored mask (red)
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[:, :, 0] = (mask > 128).astype(np.uint8) * 255  # Red channel
    
    # Blend
    overlay = cv2.addWeighted(img_rgb, 1-alpha, mask_colored, alpha, 0)
    
    return overlay

def predict_unet_from_file(filepath, models_dir="models", threshold=0.65):
    """
    Enhanced predict with crop reconstruction for accurate positioning
    """
    # 1. Load model
    models_dir = Path(models_dir)
    unet_file = models_dir / "unet_best.pt"
    if not unet_file.exists():
        candidates = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No model found in {models_dir}")
        unet_file = candidates[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet, device = load_unet_model(unet_file, device=device)
    
    # 2. Load image
    img = read_dcm_to_image(filepath)
    orig_shape = img.shape
    
    # 3. Try to find matching metadata
    patient_id, slice_index = extract_patient_slice_from_filename(filepath)
    metadata = None
    
    if patient_id and slice_index:
        metadata = find_matching_metadata(patient_id, slice_index)
        print(f"Found metadata for {patient_id} slice {slice_index}: {metadata is not None}")
    
    if metadata:
        # 4a. Use crop metadata for consistent processing
        crop_meta = metadata['crop_meta']
        
        if crop_meta['has_foreground']:
            x1, y1, x2, y2 = crop_meta['crop_bbox']
            img_crop = img[y1:y2, x1:x2]
        else:
            img_crop = img
        
        # Letterbox resize cropped region
        tensor, letterbox_meta = preprocess_img(img_crop, out_size=256)
        tensor = tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = unet(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        # Post-process
        mask_bin = postprocess_pred(probs, threshold=threshold)
        
        # Reconstruct to original coordinates
        combined_metadata = {
            'crop_meta': crop_meta,
            'letterbox_meta': letterbox_meta
        }
        mask_full = reconstruct_mask_from_crop(mask_bin, combined_metadata)
        
        print("Used crop metadata for reconstruction")
        
    else:
        # 4b. Fallback: Estimate crop region or use full image
        print("No metadata found, using estimated crop")
        
        # Option 1: Estimate pancreas region
        crop_meta = estimate_pancreas_crop(img, margin=50)
        x1, y1, x2, y2 = crop_meta['crop_bbox']
        img_crop = img[y1:y2, x1:x2]
        
        # Letterbox resize cropped region
        tensor, letterbox_meta = preprocess_img(img_crop, out_size=256)
        tensor = tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = unet(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        # Post-process
        mask_bin = postprocess_pred(probs, threshold=threshold)
        
        # Reconstruct to original coordinates
        combined_metadata = {
            'crop_meta': crop_meta,
            'letterbox_meta': letterbox_meta
        }
        mask_full = reconstruct_mask_from_crop(mask_bin, combined_metadata)
    
    # 5. Overlay
    overlay_img = overlay_mask_on_image(img, mask_full, alpha=0.4)
    pil_image = Image.fromarray(overlay_img)
    
    # 6. Calculate metrics
    vol_frac = float(mask_full.sum()) / (mask_full.shape[0] * mask_full.shape[1]) * 100
    
    meta = {
        'model_file': str(unet_file.name),
        'vol_frac_percent': vol_frac,
        'threshold': threshold,
        'has_metadata': metadata is not None,
        'orig_shape': orig_shape,
        'mask_shape': mask_full.shape,
        'patient_id': patient_id,
        'slice_index': slice_index
    }
    
    return pil_image, meta