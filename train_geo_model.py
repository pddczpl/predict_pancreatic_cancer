import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib

def threshold_tuning(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'logistic_regression': (LogisticRegression(max_iter=1000), {'C':[0.1,1,10]}),
        'random_forest': (RandomForestClassifier(random_state=42), {'n_estimators':[100,200], 'max_depth':[5,10,None]}),
        'xgboost': (XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                    {'n_estimators':[100,200],'max_depth':[3,6]}),
        'svm': (SVC(probability=True), {'C':[0.1,1,10],'kernel':['linear','rbf']})
    }

    best_models = {}
    results = []

    for name, (model, params) in models.items():
        print(f'Training {name}...')
        gs = GridSearchCV(model, params, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        
        best_model = gs.best_estimator_
        y_probs = best_model.predict_proba(X_test_scaled)[:,1]

        thr, f1 = threshold_tuning(y_test, y_probs)
        auc = roc_auc_score(y_test, y_probs)

        print(f'{name} best params: {gs.best_params_}, AUC: {auc:.4f}, Best threshold: {thr:.4f}, F1: {f1:.4f}')
        
        best_models[name] = {'model': best_model, 'scaler': scaler, 'threshold': thr}
        results.append({'name': name, 'auc': auc, 'threshold': thr, 'f1': f1})

    best_result = max(results, key=lambda x: x['auc'])
    print(f'\nBest model: {best_result["name"]} with AUC: {best_result["auc"]:.4f}')

    # Save best model and scaler and threshold
    best = best_models[best_result['name']]
    model_file = f'{best_result["name"]}_model.joblib'
    scaler_file = f'{best_result["name"]}_scaler.joblib'
    threshold_file = f'{best_result["name"]}_threshold.txt'

    joblib.dump(best['model'], model_file)
    joblib.dump(best['scaler'], scaler_file)
    with open(threshold_file, 'w') as f:
        f.write(str(best['threshold']))

    print(f'Saved model to {model_file}')
    print(f'Saved scaler to {scaler_file}')
    print(f'Saved threshold to {threshold_file}')

    return best_models, best_result
class _LazyImports:
    def load_model_and_predict(model_file, scaler_file, threshold_file, X):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        with open(threshold_file, 'r') as f:
            threshold = float(f.read().strip())
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:,1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

if __name__ == "__main__":
    df = pd.read_csv('C:\\Users\\DatGo\\OneDrive\\Documents\\Personal_Project\\predict_pancreatic_cancer\\src\\processed_features\\GSE62452_PCA_features.csv')
    df['Group'] = df['Group'].map({'t':1, 'n':0})
    y = df['Group']
    X = df.drop(columns=['Group'])

    best_models, best_result = train_and_evaluate(X, y)
    
    # Demo load model và predict lại trên test set (hoặc dữ liệu mới)
    model_file = f"{best_result['name']}_model.joblib"
    scaler_file = f"{best_result['name']}_scaler.joblib"
    threshold_file = f"{best_result['name']}_threshold.txt"

    # Lấy phần test trong original split hoặc tạo dữ liệu mới để test
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    preds, probs = load_model_and_predict(model_file, scaler_file, threshold_file, X_test)

    print('Loaded model prediction example:')
    print('Preds:', preds[:10])
    print('Probs:', probs[:10])
