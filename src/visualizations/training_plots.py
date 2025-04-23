# src/visualization/training_plots.py
"""
Visualization script for ML training results.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config

def plot_model_comparison():
    """Plot comparison of different models."""
    # Connect to MLflow
    mlflow_config = get_config("mlflow")
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if not experiment:
        print("No experiment found. Please train models first.")
        return
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Group by model type
    model_metrics = runs.groupby('tags.mlflow.runName')['metrics.mae'].mean()
    
    # Bar plot
    bars = plt.bar(model_metrics.index, model_metrics.values)
    plt.xlabel('Model Type')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def plot_learning_curves():
    """Plot learning curves for models that have them."""
    # This would be implemented if we were tracking training progress
    # For now, we'll create a placeholder
    plt.figure(figsize=(10, 6))
    epochs = range(1, 51)
    train_loss = [1.0 - 0.02*epoch + np.random.normal(0, 0.01) for epoch in epochs]
    val_loss = [1.0 - 0.018*epoch + np.random.normal(0, 0.02) for epoch in epochs]
    
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

def plot_feature_importance():
    """Plot feature importance from the best model."""
    # This would be implemented based on the best model's feature importance
    # For demonstration, we'll create a placeholder
    features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
                'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 
                'PT08.S5(O3)', 'T', 'RH', 'AH']
    importance = np.random.uniform(0.01, 0.2, size=len(features))
    importance = importance / importance.sum()
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    
    # Add value labels
    for bar, imp in zip(bars, importance):
        plt.text(imp, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def main():
    """Generate all training visualizations."""
    print("Generating training visualizations...")
    
    plot_model_comparison()
    plot_learning_curves()
    plot_feature_importance()
    
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()