import numpy as np 
from data_preprocessing import load_and_preprocess_data, interpret_pca
from model import logistic_regression_likelihood, prior
from metropolis_hastings import metropolis_hastings
from plotting import plot_trace, plot_posterior_hist, plot_autocorrelation
from diagnostics import gelman_rubin_statistic
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split

def log_posterior(beta, X, y):
    return logistic_regression_likelihood(beta, X, y) + prior(beta)

def effective_sample_size(chain):
    n = len(chain)
    if n <= 1:
        return 1
    acf = np.correlate(chain - np.mean(chain), chain - np.mean(chain), mode='full')[n-1:]
    acf /= acf[0]
    
    # Find the first negative autocorrelation
    neg_loc = np.where(acf < 0)[0]
    if len(neg_loc) > 0:
        cut_off = neg_loc[0]
    else:
        cut_off = len(acf)
    
    sum_rho = 1 + 2 * np.sum(acf[1:cut_off])
    
    return n / sum_rho

def main():
    # Load and preprocess data
    X_scaled, y, selected_features, X_pca, pca = load_and_preprocess_data('data.csv')
    
    # Interpret PCA components
    interpret_pca(pca, selected_features)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Add intercept term to training data
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    
    # Number of parameters
    num_params = X_train.shape[1]
    
    # MCMC settings
    num_chains = 4
    iterations = 20000
    burn_in = int(iterations / 2)
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Running fold {fold+1}/5")
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        # Run multiple chains
        chains = []
        acceptance_rates = []
        for chain_idx in range(num_chains):
            print(f"Running chain {chain_idx+1}/{num_chains}")
            np.random.seed(chain_idx)  # For reproducibility
            initial_beta = np.random.normal(0, 1, size=num_params)
            
            # Run Metropolis-Hastings sampling
            samples, acceptance_rate = metropolis_hastings(
                log_posterior, initial_beta, iterations, X_fold_train, y_fold_train, 
                return_acceptance=True)
            
            acceptance_rates.append(acceptance_rate)
            
            # Discard burn-in samples
            samples_burned = samples[burn_in:]
            
            chains.append(samples_burned)
        
        # Compute Gelman-Rubin statistic
        gelman_rubin_values = gelman_rubin_statistic(chains)
        
        # Parameter names
        feature_names = ['Intercept'] + [f'PC{i+1}' for i in range(num_params-1)]
        
        # Print Gelman-Rubin statistics
        print("\nGelman-Rubin Statistics:")
        for name, gr in zip(feature_names, gelman_rubin_values):
            print(f"{name}: {gr:.4f}")
        
        # Combine samples from all chains for plotting and estimation
        combined_samples = np.vstack(chains)
        
        # Plotting
        plot_trace(chains, parameter_names=feature_names, thin=10)
        plot_posterior_hist(combined_samples, parameter_names=feature_names)
        plot_autocorrelation(combined_samples, parameter_names=feature_names, thin=10)
        
        # Estimate coefficients
        beta_means = np.mean(combined_samples, axis=0)
        beta_std = np.std(combined_samples, axis=0)
        
        print("\nEstimated Coefficients:")
        for name, mean, std in zip(feature_names, beta_means, beta_std):
            print(f"{name}: {mean:.4f} ± {std:.4f}")
        
        # Compute effective sample size for each parameter
        ess = np.apply_along_axis(effective_sample_size, 0, combined_samples)
        print("\nEffective Sample Sizes:")
        for name, es in zip(feature_names, ess):
            print(f"{name}: {es:.2f}")
        
        # Predict probabilities for validation set
        z_val = X_fold_val @ beta_means
        p_val = 1 / (1 + np.exp(-z_val))
        
        # Compute AUC for this fold
        fpr, tpr, _ = roc_curve(y_fold_val, p_val)
        fold_auc = auc(fpr, tpr)
        cv_aucs.append(fold_auc)
        
        # Print acceptance rates
        for i, rate in enumerate(acceptance_rates):
            print(f"Acceptance Rate for chain {i+1}: {rate:.2f}")
    
    # Print cross-validation results
    print(f"\nCross-validation AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    
    # Evaluate on test set
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    z_test = X_test @ beta_means
    p_test = 1 / (1 + np.exp(-z_test))
    
    test_fpr, test_tpr, _ = roc_curve(y_test, p_test)
    test_auc = auc(test_fpr, test_tpr)
    
    print(f"\nTest Set AUC: {test_auc:.4f}")

    # Plot ROC curve for test set
    plt.figure()
    plt.plot(test_fpr, test_tpr, label=f'Test AUC = {test_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()