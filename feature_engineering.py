# File: src/scripts/feature_engineering.py

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import glob
import gc
import warnings
warnings.filterwarnings('ignore')


# Paths
INTEGRATED_DIR = Path("src/integrated_data")
PROCESSED_DIR = Path("src/processed_features")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load integrated data
def load_integrated_data():
    print("📊 Loading integrated data...")
    data = {}

    # GDC features & labels
    data['gdc_features'] = pd.read_csv(INTEGRATED_DIR / 'gdc' / 'gdc_features.csv', index_col=0)
    data['gdc_labels']   = pd.read_csv(INTEGRATED_DIR / 'gdc' / 'gdc_labels.csv')

    # GEO: ma trận expr_transposed, rows là mẫu (patient_id), cột là probes
    geo_df = pd.read_csv(INTEGRATED_DIR / 'geo' / 'geo_features.csv')
    # Giả sử file có cột 'patient_id' hoặc 'Sample_ID'
    if 'patient_id' in geo_df.columns:
        geo_df = geo_df.set_index('patient_id')
    elif 'Sample_ID' in geo_df.columns:
        geo_df = geo_df.set_index('Sample_ID')
    data['geo_features'] = geo_df

    data['geo_labels']   = pd.read_csv(INTEGRATED_DIR / 'geo' / 'geo_labels.csv')

    # TCIA features & labels
    data['tcia_features'] = pd.read_csv(INTEGRATED_DIR / 'tcia' / 'tcia_features.csv', index_col=0)
    data['tcia_labels']   = pd.read_csv(INTEGRATED_DIR / 'tcia' / 'tcia_labels.csv')

    # Combine labels
    labels = pd.concat([data['gdc_labels'], data['geo_labels'], data['tcia_labels']], ignore_index=True)
    labels = labels.drop_duplicates('patient_id', keep='first').reset_index(drop=True)
    data['labels'] = labels

    print(f"   ✅ Loaded: GDC {data['gdc_features'].shape}, GEO {data['geo_features'].shape}, TCIA {data['tcia_features'].shape}")
    print(f"   ✅ Total labels: {len(data['labels'])}")
    return data

# 2. Enhanced bulk gene selection with multiple strategies
def select_bulk_genes(bulk_expr, bulk_norm, bulk_rma, feature_info, labels_df, 
                      top_n=3000, strategy='variance'):
    """
    Advanced gene selection với multiple strategies
    """
    print(f"🧬 Selecting top {top_n} genes using {strategy} strategy...")
    
    # Chọn dataset tốt nhất (ưu tiên RMA > normalized > raw)
    if not bulk_rma.empty:
        expr_df = bulk_rma.set_index(bulk_rma.columns[0])
        print("   Using RMA normalized data")
    elif not bulk_norm.empty:
        expr_df = bulk_norm.set_index(bulk_norm.columns[0])
        print("   Using normalized data")
    else:
        expr_df = bulk_expr.set_index(bulk_expr.columns[0])
        print("   Using raw expression data")
        # Apply log2 transform for raw data
        expr_df = np.log2(expr_df + 1)
    
    # Filter protein-coding genes if possible
    if not feature_info.empty and 'gene_type' in feature_info.columns:
        protein_genes = feature_info.query("gene_type=='protein_coding'")['gene_id']
        available_genes = set(expr_df.index).intersection(set(protein_genes))
        if available_genes:
            expr_df = expr_df.loc[list(available_genes)]
            print(f"   Filtered to {len(expr_df)} protein-coding genes")
    
    # Gene selection strategies
    if strategy == 'variance':
        gene_scores = expr_df.var(axis=1)
        top_genes = gene_scores.nlargest(top_n).index
        
    elif strategy == 'differential':
        # Requires matching with labels
        if labels_df.empty:
            print("   ⚠️  No labels found, falling back to variance")
            gene_scores = expr_df.var(axis=1)
            top_genes = gene_scores.nlargest(top_n).index
        else:
            # Implement differential gene analysis here
            from scipy import stats
            
            # Match samples with labels
            sample_labels = labels_df.set_index('patient_id')['label']
            common_samples = set(expr_df.columns).intersection(set(sample_labels.index))
            
            if len(common_samples) > 10:
                expr_subset = expr_df[list(common_samples)]
                labels_subset = sample_labels.loc[list(common_samples)]
                
                # T-test between groups
                malignant_mask = labels_subset == 'Malignant'
                normal_mask = labels_subset.isin(['Normal', 'Benign'])
                
                if malignant_mask.sum() > 0 and normal_mask.sum() > 0:
                    pvalues = []
                    for gene in expr_subset.index:
                        malignant_vals = expr_subset.loc[gene, malignant_mask]
                        normal_vals = expr_subset.loc[gene, normal_mask]
                        if len(malignant_vals) > 1 and len(normal_vals) > 1:
                            _, pval = stats.ttest_ind(malignant_vals, normal_vals)
                            pvalues.append((gene, pval))
                    
                    pvalues.sort(key=lambda x: x[1])
                    top_genes = [gene for gene, pval in pvalues[:top_n]]
                    print(f"   Selected {len(top_genes)} differential genes")
                else:
                    gene_scores = expr_df.var(axis=1)
                    top_genes = gene_scores.nlargest(top_n).index
            else:
                gene_scores = expr_df.var(axis=1)
                top_genes = gene_scores.nlargest(top_n).index
    
    # Create feature matrix
    feature_matrix = expr_df.loc[top_genes].T
    feature_matrix.to_csv(PROCESSED_DIR / f'bulk_expression_features_{strategy}.csv')
    
    print(f"   ✅ Selected {len(top_genes)} genes, matrix shape: {feature_matrix.shape}")
    return feature_matrix

# 3. Enhanced clinical features engineering
def process_clinical_features(bulk_sample_info, bulk_meta, labels_df):
    """
    Advanced clinical features processing
    """
    print("🏥 Processing clinical features...")
    
    clinical_features = pd.DataFrame()
    
    # Process sample info
    if not bulk_sample_info.empty:
        categorical_cols = bulk_sample_info.select_dtypes(include=['object']).columns
        numerical_cols = bulk_sample_info.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0:
            # One-hot encode categorical
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(bulk_sample_info[categorical_cols])
            cat_feature_names = encoder.get_feature_names_out(categorical_cols)
            cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, 
                                index=bulk_sample_info.index)
            clinical_features = pd.concat([clinical_features, cat_df], axis=1)
        
        if len(numerical_cols) > 0:
            # Scale numerical features
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(bulk_sample_info[numerical_cols])
            num_df = pd.DataFrame(num_scaled, columns=[f"{col}_scaled" for col in numerical_cols],
                                index=bulk_sample_info.index)
            clinical_features = pd.concat([clinical_features, num_df], axis=1)
    
    # Add label information
    if not labels_df.empty:
        labels_encoded = pd.get_dummies(labels_df.set_index('patient_id')['label'], 
                                      prefix='label')
        clinical_features = clinical_features.join(labels_encoded, how='left')
    
    clinical_features.to_csv(PROCESSED_DIR / 'clinical_features.csv')
    print(f"   ✅ Clinical features shape: {clinical_features.shape}")
    return clinical_features

# 4. Enhanced TCIA imaging metadata features
def process_tcia_features(tcia_summary, tcia_modality, tcia_unified):
    """
    Comprehensive TCIA features processing - FIXED
    """
    print("🖼️  Processing TCIA imaging features...")
    
    tcia_features = pd.DataFrame()
    
    # Patient-level summary features
    if not tcia_summary.empty:
        summary_features = tcia_summary.set_index('patient_id')
        tcia_features = pd.concat([tcia_features, summary_features], axis=1)
        print(f"   Added {summary_features.shape[1]} summary features")
    
    # Modality distribution features - FIXED
    if not tcia_modality.empty:
        modality_pivot = tcia_modality.pivot_table(index='modality', values='count', 
                                                 aggfunc='sum').T
        modality_pivot.columns = [f"modality_{col}" for col in modality_pivot.columns]
        
        # FIX: Properly handle modality features
        if len(modality_pivot) > 0 and not tcia_features.empty:
            # Get the first (and only) row of modality data
            modality_row = modality_pivot.iloc[0]  # Use iloc instead of loc[0]
            
            # Add modality features to each patient
            for col in modality_pivot.columns:
                tcia_features[col] = modality_row[col]
        elif len(modality_pivot) > 0:
            # If no existing tcia_features, create from scratch
            modality_row = modality_pivot.iloc[0]
            tcia_features = pd.DataFrame([modality_row] * len(tcia_summary) if not tcia_summary.empty else [modality_row], 
                                       columns=modality_pivot.columns)
            if not tcia_summary.empty:
                tcia_features.index = tcia_summary['patient_id']
    
    # Enhanced unified metadata features
    if not tcia_unified.empty and 'patient_id' in tcia_unified.columns:
        # Aggregate by patient
        numeric_cols = tcia_unified.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            unified_agg = tcia_unified.groupby('patient_id')[numeric_cols].agg(['mean', 'std', 'count'])
            unified_agg.columns = [f"{col[0]}_{col[1]}" for col in unified_agg.columns]
            
            if tcia_features.empty:
                tcia_features = unified_agg
            else:
                tcia_features = tcia_features.join(unified_agg, how='outer')
            print(f"   Added {unified_agg.shape[1]} aggregated unified features")
    
    # Fill NaN values and ensure we have some data
    if tcia_features.empty:
        print("   ⚠️  No TCIA features generated, creating dummy dataset")
        # Create a dummy feature set if no data available
        dummy_patients = ['DUMMY_PATIENT']
        tcia_features = pd.DataFrame({'tcia_available': [0]}, index=dummy_patients)
    
    tcia_features = tcia_features.fillna(0)
    tcia_features.to_csv(PROCESSED_DIR / 'tcia_features.csv')
    print(f"   ✅ TCIA features shape: {tcia_features.shape}")
    return tcia_features

# 5. Dimensionality reduction and feature selection
def apply_dimensionality_reduction(features_df, n_components=100):
    """
    Apply PCA for dimensionality reduction - FIXED
    """
    print(f"🔄 Applying PCA with {n_components} components...")
    
    # FIX 1: Standardize column names to strings
    features_df.columns = features_df.columns.astype(str)
    
    # FIX 2: Handle non-numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_numeric = features_df[numeric_cols]
    
    print(f"   Using {len(features_numeric.columns)} numeric features for PCA")
    
    # Check if we have enough data
    if features_numeric.empty:
        print("   ⚠️  No numeric features found for PCA")
        return features_df, None
    
    if len(features_numeric) < 2:
        print("   ⚠️  Not enough samples for PCA")
        return features_df, None
    
    # FIX 3: Handle NaN and infinite values
    features_clean = features_numeric.fillna(0)
    features_clean = features_clean.replace([np.inf, -np.inf], 0)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    # Apply PCA
    n_components = min(n_components, features_scaled.shape[1], features_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create DataFrame with proper column names
    pca_columns = [f"PC_{i+1}" for i in range(features_pca.shape[1])]
    features_pca_df = pd.DataFrame(features_pca, columns=pca_columns, index=features_df.index)
    
    print(f"   ✅ PCA completed: {features_pca_df.shape}")
    print(f"   ✅ Explained variance ratio: {pca.explained_variance_ratio_[:5].sum():.3f} (first 5 components)")
    
    features_pca_df.to_csv(PROCESSED_DIR / f'features_pca_{n_components}.csv')
    return features_pca_df, pca

# 6. Main pipeline - combine all features
def run_feature_engineering_pipeline():
    print("🚀 Starting Feature Engineering Pipeline")
    data = load_integrated_data()
    for key in ['gdc_features', 'geo_features', 'tcia_features']:
        df = data[key]
        df.index = df.index.astype(str)
        data[key] = df
    data['labels']['patient_id'] = data['labels']['patient_id'].astype(str)
    # Combine feature matrices
    all_feats = data['gdc_features'] \
        .join(data['geo_features'], how='outer') \
        .join(data['tcia_features'], how='outer') \
        .fillna(0)
    all_feats.to_csv(PROCESSED_DIR / 'combined_features_raw.csv')
    print(f"   ✅ Combined features shape: {all_feats.shape}")

    # ❷ Giảm chiều
    feats_pca, _ = apply_dimensionality_reduction(all_feats, n_components=100)

    # ❸ Gắn nhãn
    final = feats_pca.join(
        data['labels'].set_index('patient_id'),
        how='left'
    )
    final.to_csv(PROCESSED_DIR / 'final_features.csv')
    print(f"🏆 Feature Engineering Complete: {final.shape}")
    return final


if __name__ == '__main__':
    run_feature_engineering_pipeline()