import os
import cv2
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
"""
python tcia/crop.py `
   --train_csv ./tcia/splits/train.csv `
   --val_csv ./tcia/splits/val.csv `
   --out_dir ./tcia/cropped `
   --margin 20 `
   --size 256
"""
def crop_with_mask_enhanced(img, mask, margin=20):
    # Tìm bounding box của foreground - SAFE VERSION
    try:
        ys, xs = np.where(mask > 0)
        
        # ✅ SAFE: Check length, not array directly
        if len(xs) == 0 or len(ys) == 0:
            crop_meta = {
                'original_shape': list(img.shape[:2]),
                'crop_bbox': [0, 0, img.shape[1], img.shape[0]],
                'has_foreground': False
            }
            return img.copy(), mask.copy(), crop_meta
        
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        
        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.shape[1], x2 + margin)
        y2 = min(img.shape[0], y2 + margin)
        
        # Crop
        img_crop = img[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2].copy()
        
        crop_meta = {
            'original_shape': list(img.shape[:2]),
            'crop_bbox': [x1, y1, x2, y2],
            'crop_shape': list(img_crop.shape[:2]),
            'has_foreground': True
        }
        
        return img_crop, mask_crop, crop_meta
        
    except Exception as e:
        print(f"Error in crop_with_mask_enhanced: {e}")
        # Return original as fallback
        crop_meta = {
            'original_shape': list(img.shape[:2]),
            'crop_bbox': [0, 0, img.shape[1], img.shape[0]],
            'has_foreground': False
        }
        return img.copy(), mask.copy(), crop_meta

def letterbox_resize_enhanced(img, mask, out_size=256):
    try:
        h, w = img.shape[:2]
        scale = min(out_size / h, out_size / w)
        nh, nw = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        
        top = (out_size - nh) // 2
        bottom = out_size - nh - top
        left = (out_size - nw) // 2
        right = out_size - nw - left
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=0)
        mask_padded = cv2.copyMakeBorder(mask_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=0)
        
        letterbox_meta = {
            'scale': scale,
            'pad': [top, bottom, left, right],
            'resized_shape': [nh, nw],
            'final_shape': [out_size, out_size]
        }
        
        return img_padded, mask_padded, letterbox_meta
        
    except Exception as e:
        print(f"Error in letterbox_resize_enhanced: {e}")
        # Return zeros as fallback
        return np.zeros((out_size, out_size), dtype=img.dtype), \
               np.zeros((out_size, out_size), dtype=mask.dtype), \
               {'scale': 1.0, 'pad': [0, 0, 0, 0], 'resized_shape': [out_size, out_size], 'final_shape': [out_size, out_size]}

def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    path = str(path).replace('\\', '/')
    
    if not os.path.exists(path):
        return None
    
    file_size = os.path.getsize(path)
    if file_size == 0:
        return None
    
    try:
        img = cv2.imread(path, flags)
        if img is None:
            return None
        return img
    except:
        return None

def process_csv_enhanced_safe(csv_path, out_dir, split_name, margin=20, out_size=256):
    df = pd.read_csv(csv_path)
    print(f"📊 Processing {split_name}: {len(df)} rows")
    
    ct_out = Path(out_dir) / split_name / "ct"
    mask_out = Path(out_dir) / split_name / "mask"
    meta_out = Path(out_dir) / split_name / "metadata"
    
    for dir_path in [ct_out, mask_out, meta_out]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    new_rows = []
    skipped = 0
    
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"Processing {i}/{len(df)} ({(i/len(df)*100):.1f}%)")
        
        try:
            ct_path = str(row['ct_path']).replace('\\', '/')
            mask_path = str(row['label_path']).replace('\\', '/')
            
            if i % 100 == 0:  # Less verbose
                print(f"  Loading: {os.path.basename(ct_path)}")
            
            ct = safe_imread(ct_path)
            mask = safe_imread(mask_path)
            
            if ct is None or mask is None:
                skipped += 1
                continue
            
            # ✅ SAFE shape check
            if len(ct.shape) < 2 or len(mask.shape) < 2:
                skipped += 1
                continue
                
            ct_h, ct_w = ct.shape[:2] 
            mask_h, mask_w = mask.shape[:2]
            
            if ct_h != mask_h or ct_w != mask_w:
                skipped += 1
                continue
            
            # Convert to grayscale
            if len(ct.shape) == 3:
                ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # ✅ SAFE processing with try-catch
            ct_crop, mask_crop, crop_meta = crop_with_mask_enhanced(ct, mask, margin=margin)
            ct_pad, mask_pad, letterbox_meta = letterbox_resize_enhanced(ct_crop, mask_crop, out_size=out_size)
            
            # Metadata
            combined_meta = {
                'crop_meta': crop_meta,
                'letterbox_meta': letterbox_meta,
                'patient_id': str(row['patient_id']),
                'slice_index': int(row['slice_index'])
            }
            
            # Save files
            patient_id = str(row['patient_id']).replace('/', '_')
            slice_idx = int(row['slice_index'])
            base = f"{patient_id}_{slice_idx:04d}"
            
            ct_path_new = str(ct_out / f"{base}.png")
            mask_path_new = str(mask_out / f"{base}.png")
            meta_path_new = str(meta_out / f"{base}.json")
            
            if cv2.imwrite(ct_path_new, ct_pad) and cv2.imwrite(mask_path_new, mask_pad):
                with open(meta_path_new, 'w') as f:
                    json.dump(combined_meta, f, indent=2)
                
                new_rows.append({
                    'patient_id': row['patient_id'],
                    'slice_index': row['slice_index'], 
                    'ct_path': ct_path_new,
                    'label_path': mask_path_new,
                    'meta_path': meta_path_new
                })
            else:
                skipped += 1
                
        except Exception as e:
            print(f"ERROR row {i}: {str(e)}")
            skipped += 1
            continue
    
    # Save CSV
    out_csv = Path(out_dir) / f"{split_name}_cropped_enhanced.csv"
    if new_rows:
        pd.DataFrame(new_rows).to_csv(out_csv, index=False)
    
    print(f"✅ {split_name}: {len(new_rows)} processed, {skipped} skipped")
    return len(new_rows), skipped

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True) 
    ap.add_argument('--out_dir', type=str, default='./cropped')
    ap.add_argument('--margin', type=int, default=20)
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()
    
    print("🚀 ULTRA-SAFE crop processing...")
    
    train_ok, train_skip = process_csv_enhanced_safe(args.train_csv, args.out_dir, 'train', 
                                                    margin=args.margin, out_size=args.size)
    val_ok, val_skip = process_csv_enhanced_safe(args.val_csv, args.out_dir, 'val',
                                                margin=args.margin, out_size=args.size)
    
    print(f"\n📈 FINAL SUMMARY:")
    print(f"Train: {train_ok} processed, {train_skip} skipped") 
    print(f"Val: {val_ok} processed, {val_skip} skipped")
    print(f"Total: {train_ok + val_ok} samples ready")