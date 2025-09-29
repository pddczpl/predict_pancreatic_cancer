# src/scripts/data_integration.py

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import dask.dataframe as dd
from dask.distributed import Client, as_completed
import sys
from tcia_data_loader import TCIADataLoader

def setup_project_structure():
    base = Path("src")
    folders = [
        base/"integrated_data",
        base/"processed_features", 
        base/"models",
        base/"results",
        base/"scripts",
        base/"datasets",
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

def load_gdc_with_dask(path):
    print("📋 Loading GDC clinical data with Dask")
    ddf = dd.read_csv(path, assume_missing=True)
    print(f"   ✅ Partitions: {ddf.npartitions}")
    return ddf

def load_geo_bulk_with_dask(expr_path, meta_path, annot_path,
                            sample_info_path, feature_info_path,
                            normalized_expr_path, rma_expr_path):
    print("🧬 Loading GSE62452 bulk RNA-seq with Dask (optimized)")
    
    # Dùng blocksize nhỏ + sample lớn để tạo partition hợp lý
    expr = dd.read_csv(expr_path, assume_missing=True, blocksize="128MB", sample=10_000_000)
    norm_expr = dd.read_csv(normalized_expr_path, assume_missing=True, blocksize="128MB", sample=10_000_000)
    rma_expr = dd.read_csv(rma_expr_path, assume_missing=True, blocksize="128MB", sample=10_000_000)
    
    meta = dd.read_csv(meta_path, assume_missing=True, sample=1_000_000)
    annot = dd.read_csv(annot_path, assume_missing=True, sample=1_000_000)
    sample_info = dd.read_csv(sample_info_path, assume_missing=True, sample=1_000_000)
    feature_info = dd.read_csv(feature_info_path, assume_missing=True, sample=1_000_000)
    
    print(f"   ✅ expr parts: {expr.npartitions}, norm: {norm_expr.npartitions}, rma: {rma_expr.npartitions}")
    return expr, meta, annot, sample_info, feature_info, norm_expr, rma_expr

def save_ddf_simple_with_progress(ddf, output_dir, base_name):
    """Version có log chi tiết để debug"""
    os.makedirs(output_dir, exist_ok=True)
    total_parts = ddf.npartitions
    print(f"   💾 Saving {base_name} with {total_parts} partitions")
    
    for idx, delayed_part in enumerate(ddf.to_delayed()):
        print(f"   🔄 Starting partition {idx+1}/{total_parts}...")
        start = time.time()
        
        print(f"   📥 Computing partition {idx+1}...")
        df = delayed_part.compute()
        compute_time = time.time() - start
        print(f"   ✅ Computed in {compute_time:.1f}s - Shape: {df.shape}")

        print(f"   🔧 Optimizing dtypes...")
        dtype_start = time.time()
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
            except:
                df[col] = df[col].astype(str)
        dtype_time = time.time() - dtype_start
        print(f"   ✅ Dtype optimization: {dtype_time:.1f}s")

        print(f"   💾 Writing to Feather...")
        write_start = time.time()
        path = os.path.join(output_dir, f"{base_name}_part{idx}.feather")
        df.reset_index().to_feather(path)
        write_time = time.time() - write_start
        
        total_time = time.time() - start
        percent = (idx + 1) / total_parts * 100
        
        print(f"   ✅ Part {idx+1}/{total_parts} DONE - Total: {total_time:.1f}s "
              f"(Compute: {compute_time:.1f}s, Write: {write_time:.1f}s) - {percent:.1f}%")
        
        del df

def load_tcia_metadata(meta_path):
    print("🏥 Loading TCIA metadata (Pandas)")
    df = pd.read_csv(meta_path, low_memory=False)
    print(f"   ✅ Records: {len(df)}")
    return df

def save_tcia_summaries(tcia_meta_df):
    print("🏷️ Generating TCIA summaries")
    
    # Tách pixel_spacing thành hai cột số
    spacing = tcia_meta_df['pixel_spacing'].astype(str).str.strip('[]').str.split(',', expand=True)
    tcia_meta_df['spacing_x'] = pd.to_numeric(spacing[0], errors='coerce')
    tcia_meta_df['spacing_y'] = pd.to_numeric(spacing[1], errors='coerce')
    
    summary = tcia_meta_df.groupby('patient_id').agg(
        ct_slices=('data_type', lambda x: (x=='CT').sum()),
        label_slices=('data_type', lambda x: (x=='LABEL').sum()),
        seg_slices=('data_type', lambda x: (x=='SEG').sum()),
        avg_thickness=('slice_thickness','mean'),
        avg_spacing_x=('spacing_x','mean'),
        avg_spacing_y=('spacing_y','mean')
    ).reset_index()
    
    summary.to_csv("src/integrated_data/tcia_patient_summary.csv", index=False)
    
    modality_dist = tcia_meta_df['data_type'].value_counts().rename_axis('modality').reset_index(name='count')
    modality_dist.to_csv("src/integrated_data/tcia_modality_distribution.csv", index=False)

def extract_labels_from_gdc(gdc_ddf):
    print("🏷️ Extracting diagnostic labels from GDC")
    
    id_col = 'cases.case_id'
    diag_col = 'diagnoses.primary_diagnosis'
    
    def map_label(dx):
        if pd.isna(dx): return 'Unknown'
        dx = str(dx).lower()
        if 'adenocarcinoma' in dx or 'cancer' in dx: return 'Malignant'
        if 'benign' in dx: return 'Benign'
        if 'normal' in dx: return 'Normal'
        return 'Unknown'
    
    # Xử lý từng partition của Dask
    def process_chunk(chunk):
        if id_col not in chunk.columns or diag_col not in chunk.columns:
            return pd.DataFrame(columns=['patient_id', 'label'])
        
        df = chunk[[id_col, diag_col]].dropna()
        df.columns = ['patient_id', 'primary_diagnosis']
        df['label'] = df['primary_diagnosis'].apply(map_label)
        
        return df[['patient_id', 'label']]
    
    # Áp dụng cho từng partition
    lab_ddf = gdc_ddf.map_partitions(process_chunk)
    lab_df = lab_ddf.compute()
    
    # Deduplicate và save
    lab_df = lab_df.drop_duplicates(subset=['patient_id'])
    
    print(f"   ✅ Label counts: {lab_df['label'].value_counts().to_dict()}")
    return lab_df

def process_geo_dataset_only():
    expr_matrix = pd.read_csv("geo/GSE62452/GSE62452_RMA_expression_matrix.csv", index_col=0)
    expr_transposed = expr_matrix.T  # Rows: samples, Columns: probes
    sample_meta = pd.read_csv("geo/GSE62452/GSE62452_sample_metadata.csv")
    gene_anno = pd.read_csv("geo/GSE62452/GSE62452_gene_annotations.csv")
    # Nếu cần annotation tích hợp:
    expr_annotated = pd.read_csv("geo/GSE62452/GSE62452_RMA_annotated_expression.csv")
    geo_labels = sample_meta[['Sample_ID', 'Tissue_Type']].copy()
    geo_labels.columns = ['patient_id', 'label']
    # Merge thêm nếu muốn mapping gene
    return geo_labels, expr_transposed, gene_anno, expr_annotated


def process_gdc_dataset_only():
    """Process GDC dataset riêng biệt với clinical labels"""
    print("🏥 Processing GDC dataset independently...")
    
    # Create GDC directory
    Path("src/integrated_data/gdc").mkdir(parents=True, exist_ok=True)
    
    # Load GDC data
    gdc_ddf = load_gdc_with_dask("gdc/final_merged.csv")
    
    # Create GDC-specific labels từ diagnosis
    gdc_labels = extract_labels_from_gdc(gdc_ddf)
    gdc_labels.to_csv("src/integrated_data/gdc/gdc_labels.csv", index=False)
    
    # Save GDC features
    gdc_features = gdc_ddf.compute()
    gdc_features.to_csv("src/integrated_data/gdc/gdc_features.csv", index=False)
    
    print(f"   ✅ GDC processed: {len(gdc_labels)} patients")
    return gdc_labels

def process_tcia_dataset_only():
    """TCIA: chỉ lấy metadata đã chuẩn hóa, lấy đúng các cột cần phân tích"""
    tcia_df = pd.read_csv("tcia/COMBINED_TCIA_WITH_DATASET_LABELS.csv")
    # Rename/chuẩn hóa cột ID nếu cần
    if 'subject_id' in tcia_df.columns:
        tcia_df['patient_id'] = tcia_df['subject_id']
    # Chỉ giữ các cột cần dùng cho downstream modeling
    selected_cols = ["patient_id", "modality", "slice_location", "slice_index", "data_type", "study_id","study_uid","series_id","series_uid", "series_number",
                     "slice_thickness", "pixel_spacing", "pancreas_ratio","model","manufacturer","sop_class_uid","sop_class_name","institution", "collection", "series_description"]
    tcia_features = tcia_df[selected_cols]
    tcia_labels = pd.DataFrame({'patient_id': tcia_df['patient_id'].unique(), 'label': 'Has_Imaging'})
    return tcia_labels, tcia_features


def run_modular_data_integration():
    print("🚀 Starting MODULAR data integration")
    print("="*60)
    
    setup_project_structure()
    
    geo_labels, geo_expr, geo_gene_anno, geo_expr_annotated = process_geo_dataset_only()
    gdc_labels = process_gdc_dataset_only()
    tcia_labels, tcia_features = process_tcia_dataset_only()
    
    # Lưu file GEO
    output_geo_dir = "src/integrated_data/geo"
    os.makedirs(output_geo_dir, exist_ok=True)
    geo_labels.to_csv(os.path.join(output_geo_dir, "geo_labels.csv"), index=False)
    geo_expr.to_csv(os.path.join(output_geo_dir, "geo_features.csv"))  # nếu cần
    
    # Lưu file TCIA
    output_tcia_dir = "src/integrated_data/tcia"
    os.makedirs(output_tcia_dir, exist_ok=True)
    tcia_labels.to_csv(os.path.join(output_tcia_dir, "tcia_labels.csv"), index=False)
    tcia_features.to_csv(os.path.join(output_tcia_dir, "tcia_features.csv"), index=False)
    
    print("\n🏆 MODULAR DATA INTEGRATION COMPLETED!")
    print("📊 Individual Dataset Summary:")
    print(f"   🧬 GEO: {len(geo_labels)} samples with {geo_labels['label'].nunique()} tissue types")
    print(f"   🏥 GDC: {len(gdc_labels)} patients with {gdc_labels['label'].nunique()} diagnosis types") 
    print(f"   🖼️  TCIA: {len(tcia_labels)} patients with imaging data")
    
    return geo_labels, gdc_labels, (tcia_labels, tcia_features)


if __name__ == "__main__":
    run_modular_data_integration()
