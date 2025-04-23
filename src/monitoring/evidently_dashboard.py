"""
Evidently monitoring dashboard for air quality prediction system.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues
from evidently.tests import TestValueRange, TestColumnDrift

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("monitoring_dashboard")

class AirQualityMonitor:
    """Monitor data drift and model performance for air quality predictions."""
    
    def __init__(self):
        """Initialize the monitor with configuration."""
        self.data_config = get_config("data")
        self.monitoring_config = get_config("monitoring")
        self.app = dash.Dash(__name__)
        self.reference_data = None
        self.current_data = None
        
    def load_data(self):
        """Load reference and current data for monitoring."""
        try:
            # Load processed data
            data_path = self.data_config["processed_data_path"]
            if data_path.exists():
                all_data = pd.read_csv(data_path)
                
                # Use first 70% as reference data
                split_index = int(len(all_data) * 0.7)
                self.reference_data = all_data.iloc[:split_index]
                self.current_data = all_data.iloc[split_index:]
                
                logger.info(f"Data loaded. Reference: {len(self.reference_data)} rows, Current: {len(self.current_data)} rows")
            else:
                logger.warning(f"Data file not found: {data_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def generate_data_drift_report(self) -> dict:
        """Generate data drift report using Evidently."""
        if self.reference_data is None or self.current_data is None:
            return None
        
        try:
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(reference_data=self.reference_data, current_data=self.current_data)
            
            return report.as_dict()
        except Exception as e:
            logger.error(f"Error generating data drift report: {e}")
            return None
    
    def generate_regression_report(self) -> dict:
        """Generate regression performance report."""
        if self.current_data is None or 'predicted_CO' not in self.current_data.columns:
            return None
        
        try:
            report = Report(metrics=[
                RegressionPreset(),
            ])
            
            current_data_with_pred = self.current_data.copy()
            current_data_with_pred['target'] = current_data_with_pred[self.data_config["target_column"]]
            current_data_with_pred['prediction'] = current_data_with_pred['predicted_CO']
            
            report.run(reference_data=None, current_data=current_data_with_pred)
            
            return report.as_dict()
        except Exception as e:
            logger.error(f"Error generating regression report: {e}")
            return None
    
    def run_tests(self) -> dict:
        """Run data quality tests."""
        if self.current_data is None:
            return None
        
        try:
            tests = TestSuite(tests=[
                TestNumberOfColumnsWithMissingValues(),
                TestNumberOfRowsWithMissingValues(),
                TestValueRange(column_name=self.data_config["target_column"], left=0, right=20),
                TestColumnDrift(column_name=self.data_config["target_column"]),
            ])
            
            tests.run(reference_data=self.reference_data, current_data=self.current_data)
            
            return tests.as_dict()
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return None
    
    def create_layout(self):
        """Create the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Air Quality Monitoring Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                html.H2("Data Quality Overview"),
                html.Div(id='data-quality-status'),
            ], style={'margin': '20px'}),
            
            html.Div([
                html.H2("Model Performance"),
                dcc.Graph(id='mae-over-time'),
                dcc.Graph(id='prediction-error-distribution'),
            ], style={'margin': '20px'}),
            
            html.Div([
                html.H2("Data Drift"),
                html.Div(id='data-drift-status'),
                dcc.Graph(id='feature-drift-chart'),
            ], style={'margin': '20px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=self.monitoring_config["monitoring_interval"] * 1000,  # Convert to milliseconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('data-quality-status', 'children'),
             Output('mae-over-time', 'figure'),
             Output('prediction-error-distribution', 'figure'),
             Output('data-drift-status', 'children'),
             Output('feature-drift-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Reload data
            self.load_data()
            
            # Data quality status
            tests_result = self.run_tests()
            if tests_result:
                test_summary = tests_result['summary']
                quality_status = html.Div([
                    html.P(f"Total tests: {test_summary['total_tests']}"),
                    html.P(f"Passed: {test_summary['success_tests']}", style={'color': 'green'}),
                    html.P(f"Failed: {test_summary['failed_tests']}", style={'color': 'red'}),
                ])
            else:
                quality_status = html.P("No test data available")
            
            # MAE over time
            if self.current_data is not None and 'prediction_error' in self.current_data.columns:
                df = self.current_data.copy()
                df['timestamp'] = pd.to_datetime(df['processed_at'] if 'processed_at' in df.columns else df.index)
                df = df.set_index('timestamp').resample('1H').mean().reset_index()
                
                mae_figure = go.Figure()
                mae_figure.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['prediction_error'],
                    mode='lines+markers',
                    name='MAE'
                ))
                mae_figure.update_layout(
                    title='Model MAE Over Time',
                    xaxis_title='Time',
                    yaxis_title='Mean Absolute Error'
                )
            else:
                mae_figure = go.Figure()
            
            # Prediction error distribution
            if self.current_data is not None and 'prediction_error' in self.current_data.columns:
                error_figure = go.Figure()
                error_figure.add_trace(go.Histogram(
                    x=self.current_data['prediction_error'],
                    nbinsx=30,
                    name='Error Distribution'
                ))
                error_figure.update_layout(
                    title='Prediction Error Distribution',
                    xaxis_title='Prediction Error',
                    yaxis_title='Count'
                )
            else:
                error_figure = go.Figure()
            
            # Data drift status
            drift_report = self.generate_data_drift_report()
            if drift_report:
                drift_summary = drift_report['metrics'][0]['result']
                drift_status = html.Div([
                    html.P(f"Number of drifted features: {drift_summary['drift_share']}"),
                    html.P(f"Dataset drift detected: {'Yes' if drift_summary['dataset_drift'] else 'No'}",
                           style={'color': 'red' if drift_summary['dataset_drift'] else 'green'}),
                ])
            else:
                drift_status = html.P("No drift data available")
            
            # Feature drift chart
            if drift_report:
                drift_data = drift_report['metrics'][0]['result']['drift_by_columns']
                drift_df = pd.DataFrame([
                    {'Feature': k, 'Drift Score': v['drift_score']}
                    for k, v in drift_data.items()
                ])
                
                drift_figure = px.bar(
                    drift_df,
                    x='Feature',
                    y='Drift Score',
                    title='Feature Drift Scores'
                )
                drift_figure.update_layout(xaxis_tickangle=-45)
            else:
                drift_figure = go.Figure()
            
            return quality_status, mae_figure, error_figure, drift_status, drift_figure
    
    def run(self):
        """Run the monitoring dashboard."""
        self.load_data()
        self.create_layout()
        self.setup_callbacks()
        
        logger.info(f"Starting monitoring dashboard on port {self.monitoring_config['dashboard_port']}")
        self.app.run_server(host='0.0.0.0', port=self.monitoring_config['dashboard_port'], debug=False)

def main():
    """Main entry point."""
    monitor = AirQualityMonitor()
    monitor.run()

if __name__ == "__main__":
    main()