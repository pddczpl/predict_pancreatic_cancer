# tcia_streaming.py
# Streaming (no-chunk, low-RAM) pipeline for TCIA Pancreas-CT
# - Lưu từng lát ngay ra đĩa (PNG hoặc NPY)
# - Ghi metadata từng dòng vào CSV
# - Có tùy chọn grayscale (1 kênh) hoặc RGB (3 kênh)

import os
import re
import gc
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import cv2


# =========================
# Utils
# =========================

def extract_patient_id_from_path(path_str):
    """Extract patient ID from various path formats."""
    pancreas_match = re.search(r'PANCREAS_(\d+)', str(path_str))
    if pancreas_match:
        return f"PANCREAS_{pancreas_match.group(1).zfill(4)}"
    label_match = re.search(r'label(\d+)', str(path_str))
    if label_match:
        return f"PANCREAS_{label_match.group(1).zfill(4)}"
    return "Unknown"


def unified_image_preprocessing(image, image_type="CT", out_size=(512, 512), out_channels=1):
    """
    Unified preprocessing cho CT/LABEL:
    - CT: window HU [-150, 250] -> scale 0..255 uint8
    - LABEL: nhị phân 0/255
    - Resize về out_size
    - out_channels: 1 (grayscale) hoặc 3 (RGB)
    """
    # Handle multi-channel input
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., 0]

    # Normalize theo loại
    if image_type == "CT":
        # Pancreas-specific windowing
        window_center, window_width = 40, 400
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image_type == "LABEL":
        if image.max() > 1:
            image = (image > 0).astype(np.uint8) * 255
        else:
            image = (image * 255).astype(np.uint8)
    else:
        # General min-max
        if image.dtype != np.uint8:
            imin, imax = float(image.min()), float(image.max())
            if imax > imin:
                image = ((image - imin) / (imax - imin) * 255.0).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    # Resize
    image_resized = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)

    if out_channels == 3:
        if image_resized.ndim == 2:
            image_resized = np.stack([image_resized]*3, axis=-1)  # H, W, 3
        elif image_resized.shape[-1] == 1:
            image_resized = np.repeat(image_resized, 3, axis=-1)
        # else: assume already 3
    else:
        # 1 channel: ensure 2D
        if image_resized.ndim == 3 and image_resized.shape[-1] == 3:
            # convert to gray just by taking first channel (đã cùng thông tin)
            image_resized = image_resized[..., 0]
        # else: leave as 2D

    return image_resized


def safe_get_dicom_attr(ds, attr, default, convert_type=str):
    """Safely get DICOM attribute with type conversion."""
    try:
        value = getattr(ds, attr, default)
        if value is None:
            return convert_type(default) if convert_type != str else str(default)
        return convert_type(value)
    except (ValueError, TypeError, AttributeError):
        return convert_type(default) if convert_type != str else str(default)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_image(file_path: Path, img: np.ndarray, save_format: str = "png"):
    save_format = save_format.lower()
    if save_format == "png":
        # cv2.imwrite chấp nhận 1 kênh (grayscale) hoặc BGR 3 kênh
        if img.ndim == 3 and img.shape[-1] == 3:
            cv2.imwrite(str(file_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(file_path), img)
    elif save_format == "npy":
        np.save(str(file_path), img)
    else:
        raise ValueError(f"Unsupported save_format: {save_format}")


def _dicom_sort_key(dcm_path: Path):
    """
    Lấy khóa sort ổn định theo DICOM tag (không đọc pixel):
    - Ưu tiên InstanceNumber
    - fallback: SliceLocation
    - cuối cùng: tên file
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            return (0, int(inst))
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            return (1, float(sl))
    except Exception:
        pass
    return (2, str(dcm_path.name))


# =========================
# Streaming processors
# =========================

def process_labels_streaming(
    labels_folder,
    output_root="PANCREAS_CT_PERFILE",
    save_format="png",
    out_size=(512, 512),
    out_channels=1
):
    """
    Đọc từng file NIfTI label, giữ các lát có tụy và SAVE NGAY từng lát.
    Ghi metadata theo dòng vào CSV. Không tích trữ RAM.
    """
    print("STREAMING LABELS (no chunk, no RAM accumulation)")
    labels_path = Path(labels_folder)
    if not labels_path.exists():
        print(f"Labels folder not found: {labels_folder}")
        return {"slices": 0, "files_ok": 0, "files_err": 0}

    out_images_root = Path(output_root) / "labels"
    _ensure_dir(out_images_root)
    meta_path = Path(output_root) / "labels_metadata.csv"

    label_files = sorted(labels_path.glob("label*.nii.gz"))
    print(f"Found {len(label_files)} label files")

    headers = [
        'dataset','patient_id','original_file','slice_index','data_type','modality',
        'has_pancreas','pancreas_pixels','total_pixels','pancreas_ratio',
        'original_shape','processed_shape','file_path','saved_path','saved_format',
        'created_at'
    ]
    first_write = not meta_path.exists()
    with open(meta_path, 'a', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=headers)
        if first_write:
            writer.writeheader()

        files_ok = files_err = total_slices = 0
        for idx, label_file in enumerate(label_files):
            try:
                nii = nib.load(label_file)
                label_data = nii.get_fdata()
                patient_id = extract_patient_id_from_path(label_file.name)
                kept = 0

                if label_data.ndim == 3:
                    patient_dir = out_images_root / patient_id
                    _ensure_dir(patient_dir)

                    for slice_idx in range(label_data.shape[2]):
                        slice_data = label_data[:, :, slice_idx]
                        if np.any(slice_data > 0):
                            processed = unified_image_preprocessing(
                                slice_data, "LABEL", out_size=out_size, out_channels=out_channels
                            )

                            stem = f"{patient_id}_label_{slice_idx:05d}"
                            ext = ".png" if save_format.lower() == "png" else ".npy"
                            save_path = patient_dir / f"{stem}{ext}"
                            _save_image(save_path, processed, save_format)

                            writer.writerow({
                                'dataset': 'TCIA-Labels',
                                'patient_id': patient_id,
                                'original_file': str(label_file.name),
                                'slice_index': slice_idx,
                                'data_type': 'LABEL',
                                'modality': 'SEG',
                                'has_pancreas': True,
                                'pancreas_pixels': int(np.sum(slice_data > 0)),
                                'total_pixels': int(slice_data.size),
                                'pancreas_ratio': float(np.sum(slice_data > 0) / slice_data.size),
                                'original_shape': str(slice_data.shape),
                                'processed_shape': str(processed.shape),
                                'file_path': str(label_file),
                                'saved_path': str(save_path),
                                'saved_format': save_format.lower(),
                                'created_at': datetime.utcnow().isoformat()
                            })
                            kept += 1
                            total_slices += 1

                            del processed
                files_ok += 1
                print(f"[{idx+1}/{len(label_files)}] {label_file.name} -> kept {kept} slices")
                del label_data, nii
            except Exception as e:
                files_err += 1
                print(f"   Error processing {label_file}: {e}")
            finally:
                gc.collect()

    print(f"Labels streaming done. slices={total_slices}, files_ok={files_ok}, files_err={files_err}")
    return {"slices": total_slices, "files_ok": files_ok, "files_err": files_err}


def process_ct_scans_streaming(
    ct_manifest_folder,
    output_root="PANCREAS_CT_PERFILE",
    save_format="png",
    out_size=(512, 512),
    out_channels=1
):
    """
    Đọc từng DICOM, xử lý và SAVE NGAY từng lát. Ghi metadata theo dòng vào CSV.
    Không giữ ảnh trong RAM và không chunk.
    """
    print("STREAMING CT (no chunk, no RAM accumulation)")
    manifest_path = Path(ct_manifest_folder)
    if not manifest_path.exists():
        print(f"CT manifest folder not found: {ct_manifest_folder}")
        return {"slices": 0, "files_ok": 0, "files_err": 0}

    out_images_root = Path(output_root) / "ct"
    _ensure_dir(out_images_root)
    meta_path = Path(output_root) / "ct_metadata.csv"

    dicom_files = list(manifest_path.rglob("*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files")

    headers = [
        'dataset','patient_id','original_file','slice_index','data_type','modality',
        'series_id','study_id','slice_location','slice_thickness','pixel_spacing',
        'study_date','series_date','institution','manufacturer','model',
        'rescale_slope','rescale_intercept','original_shape','processed_shape',
        'file_path','saved_path','saved_format','created_at'
    ]
    first_write = not meta_path.exists()
    with open(meta_path, 'a', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=headers)
        if first_write:
            writer.writeheader()

        files_ok = files_err = total_slices = 0

        # Gom theo bệnh nhân để thư mục gọn gàng
        patient_files = {}
        for dcm in dicom_files:
            pid = extract_patient_id_from_path(str(dcm))
            patient_files.setdefault(pid, []).append(dcm)

        for patient_idx, (patient_id, files) in enumerate(patient_files.items()):
            print(f"Patient {patient_idx+1}/{len(patient_files)}: {patient_id} ({len(files)} files)")

            # Sort bằng tag (đọc header nhanh, không lấy pixel)
            try:
                files.sort(key=_dicom_sort_key)
            except Exception:
                files.sort()

            patient_dir = out_images_root / patient_id
            _ensure_dir(patient_dir)

            for idx, dcm_file in enumerate(files):
                try:
                    ds = pydicom.dcmread(dcm_file)  # đọc đủ để lấy pixel
                    if not hasattr(ds, 'pixel_array'):
                        continue

                    image = ds.pixel_array.astype(float)
                    # Rescale (nếu có)
                    try:
                        slope = float(getattr(ds, 'RescaleSlope', 1.0))
                        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                    except (ValueError, TypeError):
                        slope, intercept = 1.0, 0.0
                    image = image * slope + intercept

                    processed = unified_image_preprocessing(
                        image, "CT", out_size=out_size, out_channels=out_channels
                    )

                    stem = f"{patient_id}_ct_{idx:05d}"
                    ext = ".png" if save_format.lower() == "png" else ".npy"
                    save_path = patient_dir / f"{stem}{ext}"
                    _save_image(save_path, processed, save_format)

                    meta = {
                        'dataset': 'TCIA-CT',
                        'patient_id': patient_id,
                        'original_file': str(dcm_file.name),
                        'slice_index': idx,
                        'data_type': 'CT',
                        'modality': safe_get_dicom_attr(ds, 'Modality', 'CT'),
                        'series_id': safe_get_dicom_attr(ds, 'SeriesInstanceUID', f'Series_{patient_id}_{idx}'),
                        'study_id': safe_get_dicom_attr(ds, 'StudyInstanceUID', f'Study_{patient_id}'),
                        'slice_location': safe_get_dicom_attr(ds, 'SliceLocation', idx, float),
                        'slice_thickness': safe_get_dicom_attr(ds, 'SliceThickness', 1.0, float),
                        'pixel_spacing': safe_get_dicom_attr(ds, 'PixelSpacing', '[1.0, 1.0]'),
                        'study_date': safe_get_dicom_attr(ds, 'StudyDate', 'Unknown'),
                        'series_date': safe_get_dicom_attr(ds, 'SeriesDate', 'Unknown'),
                        'institution': safe_get_dicom_attr(ds, 'InstitutionName', 'Unknown'),
                        'manufacturer': safe_get_dicom_attr(ds, 'Manufacturer', 'Unknown'),
                        'model': safe_get_dicom_attr(ds, 'ManufacturerModelName', 'Unknown'),
                        'rescale_slope': safe_get_dicom_attr(ds, 'RescaleSlope', 1.0, float),
                        'rescale_intercept': safe_get_dicom_attr(ds, 'RescaleIntercept', 0.0, float),
                        'original_shape': str(image.shape),
                        'processed_shape': str(processed.shape),
                        'file_path': str(dcm_file),
                        'saved_path': str(save_path),
                        'saved_format': save_format.lower(),
                        'created_at': datetime.utcnow().isoformat()
                    }
                    writer.writerow(meta)
                    total_slices += 1
                    files_ok += 1

                    del image, processed, ds
                except Exception as e:
                    files_err += 1
                    if files_err <= 20:
                        print(f"   Error with {dcm_file.name}: {e}")
                finally:
                    gc.collect()

    print(f"CT streaming done. slices={total_slices}, files_ok={files_ok}, files_err={files_err}")
    return {"slices": total_slices, "files_ok": files_ok, "files_err": files_err}


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("TCIA PROCESSING - STREAMING PER-FILE (no chunk)")
    print("="*60)

    # Thư mục gốc dataset (đổi nếu cần)
    base_folder = Path("./tcia")

    # Đường dẫn dataset con
    labels_folder = base_folder / "Pancreas-CT" / "TCIA_pancreas_labels-02-05-2017"
    ct_folder = base_folder / "Pancreas-CT" / "manifest-1599750808610" / "Pancreas-CT"

    # Cấu hình đầu ra
    output_root = base_folder / "PANCREAS_CT_PERFILE"
    _ensure_dir(Path(output_root))

    # Lựa chọn định dạng & kênh
    # - save_format: "png" (nhỏ, xem nhanh) hoặc "npy" (đọc nhanh trong Python)
    # - out_channels: 1 (grayscale) hoặc 3 (RGB)
    save_format = "png"
    out_channels = 1     # đổi 3 nếu muốn RGB
    out_size = (512, 512)

    try:
        print("\n1) Streaming Pancreas LABELS...")
        labels_stats = process_labels_streaming(
            labels_folder,
            output_root=output_root,
            save_format=save_format,
            out_size=out_size,
            out_channels=out_channels
        )

        print("\n2) Streaming Pancreas CT SCANS...")
        ct_stats = process_ct_scans_streaming(
            ct_folder,
            output_root=output_root,
            save_format=save_format,
            out_size=out_size,
            out_channels=out_channels
        )

        # Tổng kết
        print("\n🎉 STREAMING COMPLETED!")
        print("="*60)
        print(f"Labels: {labels_stats}")
        print(f"CT:     {ct_stats}")
        print("📁 Outputs at:", output_root)
        print(f"   - {output_root}/labels/...")
        print(f"   - {output_root}/ct/...")
        print(f"   - {output_root}/labels_metadata.csv")
        print(f"   - {output_root}/ct_metadata.csv")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
