import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def apply_pca(X, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Drop 'id' column and the unnamed column
    data = data.drop(columns=['id', 'Unnamed: 32'])

    # Encode 'diagnosis' column
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features and target
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # Check for high correlations
    correlation_matrix = X.corr()
    high_corr = np.where(np.abs(correlation_matrix) > 0.8)
    high_corr = [(correlation_matrix.index[x], correlation_matrix.columns[y]) 
                 for x, y in zip(*high_corr) if x != y and x < y]
    if high_corr:
        print("High correlation between")
        for i, j in high_corr:
            print(f"{i} and {j}")
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA
    X_pca, pca = apply_pca(X_scaled)
    
    return X_scaled, y.values, selected_features, X_pca, pca

def interpret_pca(pca, feature_names, n_components=5):
    for i in range(min(n_components, pca.n_components_)):
        print(f"PC{i+1} composition:")
        sorted_features = sorted(zip(feature_names, pca.components_[i]), key=lambda x: abs(x[1]), reverse=True)
        for feature, weight in sorted_features[:5]:  # Top 5 contributors
            print(f"  {feature}: {weight:.3f}")
        print()