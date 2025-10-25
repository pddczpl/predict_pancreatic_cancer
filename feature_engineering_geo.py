import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def feature_engineering_ml_ready(input_csv_path, output_csv_path, n_components=50):
    # Đọc dữ liệu ML-ready format
    df = pd.read_csv(input_csv_path)

    # Tách nhãn và features
    labels = df['Group'].str.lower()
    features = df.drop(columns=['Sample_ID', 'Group'])

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA giảm chiều
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Đưa về DataFrame kết hợp nhãn
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Group'] = labels.values

    # Lưu ra file CSV
    pca_df.to_csv(output_csv_path, index=False)
    print(f"PCA features saved to {output_csv_path}")

# Thay đổi đường dẫn đến file của bạn
input_csv = r"C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_ML_ready_features.csv"
output_csv = r"C:\Users\DatGo\OneDrive\Documents\Personal_Project\predict_pancreatic_cancer\src\processed_features\GSE62452_PCA_features.csv"

# Thực thi feature engineering
feature_engineering_ml_ready(input_csv, output_csv)
