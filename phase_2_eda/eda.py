import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
def load_air_quality_data(filepath):
    """
    Load air quality data from CSV or Excel file
    """
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Parse datetime with a specific format
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        
        return df
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        raise

# Preprocess the data
def preprocess_data(df):
    """
    Clean and prepare the data for analysis
    """
    # Drop unnamed columns
    df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Set datetime as index
    df.set_index('Datetime', inplace=True)
    
    return df

# Basic Time Series Plots
def plot_time_series(df, columns):
    """
    Create time series plots for specified columns
    """
    plt.figure(figsize=(15,10))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.title('Pollutant Concentrations Over Time')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pollutant_time_series.png')
    plt.close()

# Correlation Heatmap
def plot_correlation_heatmap(df, columns):
    """
    Create correlation heatmap between pollutants
    """
    # Select only numeric columns
    corr_df = df[columns]
    
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Pollutants')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Descriptive Statistics
def generate_descriptive_stats(df, columns):
    """
    Generate and print descriptive statistics
    """
    print("\nDescriptive Statistics:")
    print(df[columns].describe())

# Temporal Patterns
def plot_temporal_patterns(df, columns):
    """
    Plot average concentrations by hour and day of week
    """
    # Hourly average
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    hourly_avg = df.groupby(df.index.hour)[columns].mean()
    hourly_avg.plot(kind='bar', ax=plt.gca())
    plt.title('Average Pollutant Concentrations by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Concentration')
    plt.xticks(rotation=45)

    # Day of week average
    plt.subplot(1,2,2)
    daily_avg = df.groupby(df.index.dayofweek)[columns].mean()
    daily_avg.plot(kind='bar', ax=plt.gca())
    plt.title('Average Pollutant Concentrations by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Concentration')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('temporal_patterns.png')
    plt.close()

# Boxplot of Pollutant Distributions
def plot_pollutant_distributions(df, columns):
    """
    Create boxplots to show distribution of pollutant concentrations
    """
    plt.figure(figsize=(12,6))
    df[columns].boxplot()
    plt.title('Distribution of Pollutant Concentrations')
    plt.ylabel('Concentration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pollutant_distributions.png')
    plt.close()

# Additional analysis of temporal range
def analyze_temporal_range(df):
    """
    Analyze the temporal range of the dataset
    """
    print("\nTemporal Range Analysis:")
    print(f"Start Date: {df.index.min()}")
    print(f"End Date: {df.index.max()}")
    print(f"Total Duration: {df.index.max() - df.index.min()}")
    print(f"Total Number of Measurements: {len(df)}")

# Main analysis function
def perform_air_quality_eda(filepath):
    """
    Perform complete EDA on air quality data
    """
    # Load and preprocess data
    df = load_air_quality_data(filepath)
    df = preprocess_data(df)
    
    # Select pollutant columns
    pollutant_columns = [
        'CO(GT)', 'PT08.S1(CO)', 
        'NOx(GT)', 'PT08.S3(NOx)', 
        'C6H6(GT)', 'NO2(GT)'
    ]
    
    # Ensure selected columns exist in the dataframe
    pollutant_columns = [col for col in pollutant_columns if col in df.columns]
    
    print("\nIdentified Pollutant Columns:", pollutant_columns)
    
    # Perform analyses
    generate_descriptive_stats(df, pollutant_columns)
    analyze_temporal_range(df)
    
    # Plotting
    plot_time_series(df, pollutant_columns)
    plot_correlation_heatmap(df, pollutant_columns)
    plot_temporal_patterns(df, pollutant_columns)
    plot_pollutant_distributions(df, pollutant_columns)
    
    print("EDA complete. Check generated image files.")
    
    return df

# Run the analysis
if __name__ == "__main__":
    # Use the current directory to find the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        exit(1)
    
    # Use the first CSV file found
    filepath = os.path.join(current_dir, csv_files[0])
    print(f"Analyzing file: {filepath}")
    
    perform_air_quality_eda(filepath)