"""
TIBEDO Quantum Benchmark Visualization Module

This module provides visualization tools for quantum benchmarking results,
enabling clear presentation and analysis of performance metrics for different
quantum optimization techniques.

Key components:
1. BenchmarkVisualizer: Creates visualizations of benchmark results
2. ComparisonPlotter: Generates comparative plots for different optimization methods
3. PerformanceDashboard: Creates interactive dashboards for performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import interactive visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly not found. Interactive visualizations will not be available.")

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    logger.warning("Dash not found. Interactive dashboards will not be available.")


class BenchmarkVisualizer:
    """
    Creates visualizations of quantum benchmark results.
    
    This class provides methods for generating various types of visualizations
    to help analyze and present the results of quantum benchmarking experiments.
    """
    
    def __init__(self, 
                 output_dir: str = 'benchmark_results',
                 style: str = 'seaborn-whitegrid',
                 use_interactive: bool = True,
                 fig_size: tuple = (12, 8),
                 dpi: int = 100):
        """
        Initialize the Benchmark Visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            style: Matplotlib style to use for plots
            use_interactive: Whether to use interactive visualizations when available
            fig_size: Default figure size for plots
            dpi: Resolution for saved figures
        """
        self.output_dir = output_dir
        self.style = style
        self.use_interactive = use_interactive and HAS_PLOTLY
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use(style)
    
    def plot_gate_count_comparison(self, 
                                  benchmark_results: Dict[str, Dict[str, Any]],
                                  title: str = 'Gate Count Comparison',
                                  save_path: Optional[str] = None) -> None:
        """
        Create a bar chart comparing gate counts before and after optimization.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Extract data
        methods = list(benchmark_results.keys())
        original_counts = [results.get('original_gate_count', 0) for results in benchmark_results.values()]
        optimized_counts = [results.get('optimized_gate_count', 0) for results in benchmark_results.values()]
        
        # Create figure
        plt.figure(figsize=self.fig_size)
        
        # Set width of bars
        bar_width = 0.35
        
        # Set position of bars on x axis
        r1 = np.arange(len(methods))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        plt.bar(r1, original_counts, width=bar_width, label='Original', color='skyblue')
        plt.bar(r2, optimized_counts, width=bar_width, label='Optimized', color='lightgreen')
        
        # Add labels and title
        plt.xlabel('Optimization Method')
        plt.ylabel('Gate Count')
        plt.title(title)
        plt.xticks([r + bar_width/2 for r in range(len(methods))], methods)
        plt.legend()
        
        # Add reduction percentages
        for i in range(len(methods)):
            if original_counts[i] > 0:
                reduction = (original_counts[i] - optimized_counts[i]) / original_counts[i] * 100
                plt.text(r2[i], optimized_counts[i] + 5, f"{reduction:.1f}%", 
                         ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"gate_count_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Gate count comparison plot saved to {save_path}")
        
        # Create interactive version if enabled
        if self.use_interactive:
            self._create_interactive_gate_count_plot(benchmark_results, title, save_path)
    
    def _create_interactive_gate_count_plot(self,
                                           benchmark_results: Dict[str, Dict[str, Any]],
                                           title: str,
                                           save_path: str) -> None:
        """Create an interactive version of the gate count comparison plot."""
        if not HAS_PLOTLY:
            return
        
        # Extract data
        methods = list(benchmark_results.keys())
        original_counts = [results.get('original_gate_count', 0) for results in benchmark_results.values()]
        optimized_counts = [results.get('optimized_gate_count', 0) for results in benchmark_results.values()]
        
        # Create dataframe
        df = pd.DataFrame({
            'Method': methods + methods,
            'Gate Count': original_counts + optimized_counts,
            'Type': ['Original'] * len(methods) + ['Optimized'] * len(methods)
        })
        
        # Create figure
        fig = px.bar(df, x='Method', y='Gate Count', color='Type',
                    barmode='group', title=title,
                    color_discrete_map={'Original': 'skyblue', 'Optimized': 'lightgreen'})
        
        # Add reduction percentages
        for i, method in enumerate(methods):
            if original_counts[i] > 0:
                reduction = (original_counts[i] - optimized_counts[i]) / original_counts[i] * 100
                fig.add_annotation(
                    x=method,
                    y=optimized_counts[i],
                    text=f"{reduction:.1f}%",
                    showarrow=False,
                    yshift=10
                )
        
        # Save as HTML
        interactive_save_path = save_path.replace('.png', '_interactive.html')
        fig.write_html(interactive_save_path)
        
        logger.info(f"Interactive gate count comparison plot saved to {interactive_save_path}")
    
    def plot_circuit_depth_comparison(self,
                                     benchmark_results: Dict[str, Dict[str, Any]],
                                     title: str = 'Circuit Depth Comparison',
                                     save_path: Optional[str] = None) -> None:
        """
        Create a bar chart comparing circuit depths before and after optimization.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Extract data
        methods = list(benchmark_results.keys())
        original_depths = [results.get('original_depth', 0) for results in benchmark_results.values()]
        optimized_depths = [results.get('optimized_depth', 0) for results in benchmark_results.values()]
        
        # Create figure
        plt.figure(figsize=self.fig_size)
        
        # Set width of bars
        bar_width = 0.35
        
        # Set position of bars on x axis
        r1 = np.arange(len(methods))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        plt.bar(r1, original_depths, width=bar_width, label='Original', color='coral')
        plt.bar(r2, optimized_depths, width=bar_width, label='Optimized', color='lightblue')
        
        # Add labels and title
        plt.xlabel('Optimization Method')
        plt.ylabel('Circuit Depth')
        plt.title(title)
        plt.xticks([r + bar_width/2 for r in range(len(methods))], methods)
        plt.legend()
        
        # Add reduction percentages
        for i in range(len(methods)):
            if original_depths[i] > 0:
                reduction = (original_depths[i] - optimized_depths[i]) / original_depths[i] * 100
                plt.text(r2[i], optimized_depths[i] + 2, f"{reduction:.1f}%", 
                         ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"circuit_depth_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Circuit depth comparison plot saved to {save_path}")
        
        # Create interactive version if enabled
        if self.use_interactive:
            self._create_interactive_depth_plot(benchmark_results, title, save_path)
    
    def _create_interactive_depth_plot(self,
                                      benchmark_results: Dict[str, Dict[str, Any]],
                                      title: str,
                                      save_path: str) -> None:
        """Create an interactive version of the circuit depth comparison plot."""
        if not HAS_PLOTLY:
            return
        
        # Extract data
        methods = list(benchmark_results.keys())
        original_depths = [results.get('original_depth', 0) for results in benchmark_results.values()]
        optimized_depths = [results.get('optimized_depth', 0) for results in benchmark_results.values()]
        
        # Create dataframe
        df = pd.DataFrame({
            'Method': methods + methods,
            'Circuit Depth': original_depths + optimized_depths,
            'Type': ['Original'] * len(methods) + ['Optimized'] * len(methods)
        })
        
        # Create figure
        fig = px.bar(df, x='Method', y='Circuit Depth', color='Type',
                    barmode='group', title=title,
                    color_discrete_map={'Original': 'coral', 'Optimized': 'lightblue'})
        
        # Add reduction percentages
        for i, method in enumerate(methods):
            if original_depths[i] > 0:
                reduction = (original_depths[i] - optimized_depths[i]) / original_depths[i] * 100
                fig.add_annotation(
                    x=method,
                    y=optimized_depths[i],
                    text=f"{reduction:.1f}%",
                    showarrow=False,
                    yshift=10
                )
        
        # Save as HTML
        interactive_save_path = save_path.replace('.png', '_interactive.html')
        fig.write_html(interactive_save_path)
        
        logger.info(f"Interactive circuit depth comparison plot saved to {interactive_save_path}")
    
    def plot_optimization_time_comparison(self,
                                         benchmark_results: Dict[str, Dict[str, Any]],
                                         title: str = 'Optimization Time Comparison',
                                         save_path: Optional[str] = None) -> None:
        """
        Create a bar chart comparing optimization times for different methods.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Extract data
        methods = list(benchmark_results.keys())
        times = [results.get('optimization_time', 0) for results in benchmark_results.values()]
        
        # Create figure
        plt.figure(figsize=self.fig_size)
        
        # Create bars
        bars = plt.bar(methods, times, color='lightseagreen')
        
        # Add labels and title
        plt.xlabel('Optimization Method')
        plt.ylabel('Time (seconds)')
        plt.title(title)
        
        # Add time values above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.3f}s", ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"optimization_time_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Optimization time comparison plot saved to {save_path}")
        
        # Create interactive version if enabled
        if self.use_interactive:
            self._create_interactive_time_plot(benchmark_results, title, save_path)
    
    def _create_interactive_time_plot(self,
                                     benchmark_results: Dict[str, Dict[str, Any]],
                                     title: str,
                                     save_path: str) -> None:
        """Create an interactive version of the optimization time comparison plot."""
        if not HAS_PLOTLY:
            return
        
        # Extract data
        methods = list(benchmark_results.keys())
        times = [results.get('optimization_time', 0) for results in benchmark_results.values()]
        
        # Create figure
        fig = px.bar(x=methods, y=times, title=title,
                    labels={'x': 'Optimization Method', 'y': 'Time (seconds)'})
        
        # Add time values above bars
        for i, method in enumerate(methods):
            fig.add_annotation(
                x=method,
                y=times[i],
                text=f"{times[i]:.3f}s",
                showarrow=False,
                yshift=10
            )
        
        # Save as HTML
        interactive_save_path = save_path.replace('.png', '_interactive.html')
        fig.write_html(interactive_save_path)
        
        logger.info(f"Interactive optimization time comparison plot saved to {interactive_save_path}")
    
    def plot_performance_radar(self,
                              benchmark_results: Dict[str, Dict[str, Any]],
                              metrics: List[str] = ['gate_reduction', 'depth_reduction', 'optimization_time'],
                              title: str = 'Performance Radar Chart',
                              save_path: Optional[str] = None) -> None:
        """
        Create a radar chart comparing different optimization methods across multiple metrics.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            metrics: List of metrics to include in the radar chart
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Extract data and calculate normalized metrics
        methods = list(benchmark_results.keys())
        
        # Calculate derived metrics
        normalized_metrics = {}
        for metric in metrics:
            if metric == 'gate_reduction':
                values = []
                for results in benchmark_results.values():
                    orig = results.get('original_gate_count', 0)
                    opt = results.get('optimized_gate_count', 0)
                    if orig > 0:
                        values.append((orig - opt) / orig * 100)
                    else:
                        values.append(0)
            elif metric == 'depth_reduction':
                values = []
                for results in benchmark_results.values():
                    orig = results.get('original_depth', 0)
                    opt = results.get('optimized_depth', 0)
                    if orig > 0:
                        values.append((orig - opt) / orig * 100)
                    else:
                        values.append(0)
            elif metric == 'optimization_time':
                # For time, lower is better, so we invert the normalization
                times = [results.get('optimization_time', 0) for results in benchmark_results.values()]
                max_time = max(times) if max(times) > 0 else 1
                values = [100 * (1 - time / max_time) for time in times]
            else:
                # For other metrics, just use the raw values
                values = [results.get(metric, 0) for results in benchmark_results.values()]
            
            normalized_metrics[metric] = values
        
        # Create radar chart
        # Number of variables
        N = len(metrics)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics, size=10)
        
        # Draw the chart for each method
        for i, method in enumerate(methods):
            values = [normalized_metrics[metric][i] for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(title)
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"performance_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Performance radar chart saved to {save_path}")
        
        # Create interactive version if enabled
        if self.use_interactive:
            self._create_interactive_radar_plot(benchmark_results, metrics, title, save_path)
    
    def _create_interactive_radar_plot(self,
                                      benchmark_results: Dict[str, Dict[str, Any]],
                                      metrics: List[str],
                                      title: str,
                                      save_path: str) -> None:
        """Create an interactive version of the radar chart."""
        if not HAS_PLOTLY:
            return
        
        # Extract data and calculate normalized metrics
        methods = list(benchmark_results.keys())
        
        # Calculate derived metrics
        normalized_metrics = {}
        for metric in metrics:
            if metric == 'gate_reduction':
                values = []
                for results in benchmark_results.values():
                    orig = results.get('original_gate_count', 0)
                    opt = results.get('optimized_gate_count', 0)
                    if orig > 0:
                        values.append((orig - opt) / orig * 100)
                    else:
                        values.append(0)
            elif metric == 'depth_reduction':
                values = []
                for results in benchmark_results.values():
                    orig = results.get('original_depth', 0)
                    opt = results.get('optimized_depth', 0)
                    if orig > 0:
                        values.append((orig - opt) / orig * 100)
                    else:
                        values.append(0)
            elif metric == 'optimization_time':
                # For time, lower is better, so we invert the normalization
                times = [results.get('optimization_time', 0) for results in benchmark_results.values()]
                max_time = max(times) if max(times) > 0 else 1
                values = [100 * (1 - time / max_time) for time in times]
            else:
                # For other metrics, just use the raw values
                values = [results.get(metric, 0) for results in benchmark_results.values()]
            
            normalized_metrics[metric] = values
        
        # Create radar chart
        fig = go.Figure()
        
        # Add traces for each method
        for i, method in enumerate(methods):
            values = [normalized_metrics[metric][i] for metric in metrics]
            values.append(values[0])  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],  # Close the loop
                fill='toself',
                name=method
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title=title
        )
        
        # Save as HTML
        interactive_save_path = save_path.replace('.png', '_interactive.html')
        fig.write_html(interactive_save_path)
        
        logger.info(f"Interactive radar chart saved to {interactive_save_path}")


class ComparisonPlotter:
    """
    Generates comparative plots for different quantum optimization methods.
    
    This class provides methods for creating side-by-side comparisons of
    different quantum circuit optimization techniques across various metrics.
    """
    
    def __init__(self, 
                 output_dir: str = 'comparison_results',
                 style: str = 'seaborn-whitegrid',
                 fig_size: tuple = (14, 10),
                 dpi: int = 100):
        """
        Initialize the Comparison Plotter.
        
        Args:
            output_dir: Directory to save visualization outputs
            style: Matplotlib style to use for plots
            fig_size: Default figure size for plots
            dpi: Resolution for saved figures
        """
        self.output_dir = output_dir
        self.style = style
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use(style)
    
    def plot_scaling_comparison(self,
                               benchmark_results: Dict[str, Dict[int, Dict[str, Any]]],
                               metric: str = 'optimization_time',
                               x_label: str = 'Circuit Width (qubits)',
                               y_label: Optional[str] = None,
                               title: str = 'Scaling Comparison',
                               save_path: Optional[str] = None) -> None:
        """
        Create a line plot comparing how different methods scale with circuit size.
        
        Args:
            benchmark_results: Nested dictionary of benchmark results by method and circuit size
            metric: The metric to plot
            x_label: Label for the x-axis
            y_label: Label for the y-axis (if None, will use metric name)
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Create figure
        plt.figure(figsize=self.fig_size)
        
        # Plot data for each method
        for method, results in benchmark_results.items():
            # Sort by circuit size
            sizes = sorted(results.keys())
            values = [results[size].get(metric, 0) for size in sizes]
            
            # Plot line
            plt.plot(sizes, values, marker='o', label=method)
        
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label if y_label else metric)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Use log scale if values span multiple orders of magnitude
        all_values = [results[size].get(metric, 0) 
                     for results in benchmark_results.values() 
                     for size in results.keys()]
        if max(all_values) / (min([v for v in all_values if v > 0]) or 1) > 100:
            plt.yscale('log')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"scaling_comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Scaling comparison plot saved to {save_path}")
    
    def plot_heatmap_comparison(self,
                               benchmark_results: Dict[str, Dict[str, float]],
                               title: str = 'Performance Heatmap',
                               save_path: Optional[str] = None) -> None:
        """
        Create a heatmap comparing different methods across multiple metrics.
        
        Args:
            benchmark_results: Dictionary of benchmark results by method and metric
            title: Plot title
            save_path: Path to save the figure (if None, will use default path)
        """
        # Convert to DataFrame
        methods = list(benchmark_results.keys())
        metrics = set()
        for result in benchmark_results.values():
            metrics.update(result.keys())
        metrics = list(metrics)
        
        data = []
        for method in methods:
            row = []
            for metric in metrics:
                row.append(benchmark_results[method].get(metric, 0))
            data.append(row)
        
        df = pd.DataFrame(data, index=methods, columns=metrics)
        
        # Create figure
        plt.figure(figsize=self.fig_size)
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f')
        
        # Add title
        plt.title(title)
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"heatmap_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Heatmap comparison plot saved to {save_path}")


class PerformanceDashboard:
    """
    Creates interactive dashboards for quantum performance analysis.
    
    This class provides methods for creating interactive dashboards that allow
    users to explore and analyze quantum benchmarking results.
    """
    
    def __init__(self, 
                 output_dir: str = 'dashboard_results',
                 port: int = 8050):
        """
        Initialize the Performance Dashboard.
        
        Args:
            output_dir: Directory to save dashboard outputs
            port: Port to run the dashboard server on
        """
        self.output_dir = output_dir
        self.port = port
        
        # Check if Dash is available
        if not HAS_DASH:
            logger.warning("Dash not found. Interactive dashboards will not be available.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_dashboard(self, benchmark_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create an interactive dashboard for exploring benchmark results.
        
        Args:
            benchmark_results: Dictionary of benchmark results
        """
        if not HAS_DASH:
            logger.warning("Dash not found. Interactive dashboards will not be available.")
            return
        
        # Save benchmark results to JSON for the dashboard to load
        results_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(benchmark_results, f)
        
        # Create dashboard app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1("TIBEDO Quantum Benchmarking Dashboard"),
            
            html.Div([
                html.H3("Select Metrics to Display"),
                dcc.Dropdown(
                    id='metric-selector',
                    options=[
                        {'label': 'Gate Count Reduction', 'value': 'gate_reduction'},
                        {'label': 'Circuit Depth Reduction', 'value': 'depth_reduction'},
                        {'label': 'Optimization Time', 'value': 'optimization_time'}
                    ],
                    value='gate_reduction',
                    multi=False
                )
            ]),
            
            html.Div([
                dcc.Graph(id='main-graph')
            ]),
            
            html.Div([
                html.H3("Performance Comparison"),
                dcc.Graph(id='radar-chart')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            Output('main-graph', 'figure'),
            [Input('metric-selector', 'value')]
        )
        def update_main_graph(selected_metric):
            # Load benchmark results
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create figure based on selected metric
            if selected_metric == 'gate_reduction':
                methods = list(results.keys())
                original_counts = [results[m].get('original_gate_count', 0) for m in methods]
                optimized_counts = [results[m].get('optimized_gate_count', 0) for m in methods]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=methods, y=original_counts, name='Original', marker_color='skyblue'))
                fig.add_trace(go.Bar(x=methods, y=optimized_counts, name='Optimized', marker_color='lightgreen'))
                
                fig.update_layout(
                    title='Gate Count Comparison',
                    xaxis_title='Optimization Method',
                    yaxis_title='Gate Count',
                    barmode='group'
                )
                
                # Add reduction percentages
                for i, method in enumerate(methods):
                    if original_counts[i] > 0:
                        reduction = (original_counts[i] - optimized_counts[i]) / original_counts[i] * 100
                        fig.add_annotation(
                            x=method,
                            y=optimized_counts[i],
                            text=f"{reduction:.1f}%",
                            showarrow=False,
                            yshift=10
                        )
            
            elif selected_metric == 'depth_reduction':
                methods = list(results.keys())
                original_depths = [results[m].get('original_depth', 0) for m in methods]
                optimized_depths = [results[m].get('optimized_depth', 0) for m in methods]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=methods, y=original_depths, name='Original', marker_color='coral'))
                fig.add_trace(go.Bar(x=methods, y=optimized_depths, name='Optimized', marker_color='lightblue'))
                
                fig.update_layout(
                    title='Circuit Depth Comparison',
                    xaxis_title='Optimization Method',
                    yaxis_title='Circuit Depth',
                    barmode='group'
                )
                
                # Add reduction percentages
                for i, method in enumerate(methods):
                    if original_depths[i] > 0:
                        reduction = (original_depths[i] - optimized_depths[i]) / original_depths[i] * 100
                        fig.add_annotation(
                            x=method,
                            y=optimized_depths[i],
                            text=f"{reduction:.1f}%",
                            showarrow=False,
                            yshift=10
                        )
            
            elif selected_metric == 'optimization_time':
                methods = list(results.keys())
                times = [results[m].get('optimization_time', 0) for m in methods]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=methods, y=times, marker_color='lightseagreen'))
                
                fig.update_layout(
                    title='Optimization Time Comparison',
                    xaxis_title='Optimization Method',
                    yaxis_title='Time (seconds)'
                )
                
                # Add time values
                for i, method in enumerate(methods):
                    fig.add_annotation(
                        x=method,
                        y=times[i],
                        text=f"{times[i]:.3f}s",
                        showarrow=False,
                        yshift=10
                    )
            
            else:
                fig = go.Figure()
            
            return fig
        
        @app.callback(
            Output('radar-chart', 'figure'),
            [Input('metric-selector', 'value')]
        )
        def update_radar_chart(_):
            # Load benchmark results
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Define metrics for radar chart
            metrics = ['gate_reduction', 'depth_reduction', 'optimization_time']
            methods = list(results.keys())
            
            # Calculate derived metrics
            normalized_metrics = {}
            for metric in metrics:
                if metric == 'gate_reduction':
                    values = []
                    for method in methods:
                        orig = results[method].get('original_gate_count', 0)
                        opt = results[method].get('optimized_gate_count', 0)
                        if orig > 0:
                            values.append((orig - opt) / orig * 100)
                        else:
                            values.append(0)
                elif metric == 'depth_reduction':
                    values = []
                    for method in methods:
                        orig = results[method].get('original_depth', 0)
                        opt = results[method].get('optimized_depth', 0)
                        if orig > 0:
                            values.append((orig - opt) / orig * 100)
                        else:
                            values.append(0)
                elif metric == 'optimization_time':
                    # For time, lower is better, so we invert the normalization
                    times = [results[method].get('optimization_time', 0) for method in methods]
                    max_time = max(times) if max(times) > 0 else 1
                    values = [100 * (1 - time / max_time) for time in times]
                else:
                    # For other metrics, just use the raw values
                    values = [results[method].get(metric, 0) for method in methods]
                
                normalized_metrics[metric] = values
            
            # Create radar chart
            fig = go.Figure()
            
            # Add traces for each method
            for i, method in enumerate(methods):
                values = [normalized_metrics[metric][i] for metric in metrics]
                values.append(values[0])  # Close the loop
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],  # Close the loop
                    fill='toself',
                    name=method
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title='Performance Radar Chart'
            )
            
            return fig
        
        # Run the dashboard
        logger.info(f"Starting dashboard server on port {self.port}")
        app.run_server(debug=True, port=self.port)


# Example usage
if __name__ == "__main__":
    # Create sample benchmark results
    benchmark_results = {
        'Standard': {
            'original_gate_count': 100,
            'optimized_gate_count': 80,
            'original_depth': 50,
            'optimized_depth': 40,
            'optimization_time': 0.5
        },
        'TensorNetwork': {
            'original_gate_count': 100,
            'optimized_gate_count': 60,
            'original_depth': 50,
            'optimized_depth': 30,
            'optimization_time': 1.2
        },
        'Cyclotomic': {
            'original_gate_count': 100,
            'optimized_gate_count': 70,
            'original_depth': 50,
            'optimized_depth': 35,
            'optimization_time': 0.8
        }
    }
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(output_dir='benchmark_results')
    
    # Create plots
    visualizer.plot_gate_count_comparison(benchmark_results)
    visualizer.plot_circuit_depth_comparison(benchmark_results)
    visualizer.plot_optimization_time_comparison(benchmark_results)
    visualizer.plot_performance_radar(benchmark_results)
    
    # Create comparison plotter
    comparison_plotter = ComparisonPlotter(output_dir='comparison_results')
    
    # Create scaling comparison
    scaling_results = {
        'Standard': {
            2: {'optimization_time': 0.1},
            4: {'optimization_time': 0.3},
            8: {'optimization_time': 0.7},
            16: {'optimization_time': 1.5}
        },
        'TensorNetwork': {
            2: {'optimization_time': 0.2},
            4: {'optimization_time': 0.5},
            8: {'optimization_time': 1.0},
            16: {'optimization_time': 2.0}
        },
        'Cyclotomic': {
            2: {'optimization_time': 0.15},
            4: {'optimization_time': 0.4},
            8: {'optimization_time': 0.9},
            16: {'optimization_time': 1.8}
        }
    }
    
    comparison_plotter.plot_scaling_comparison(
        scaling_results,
        metric='optimization_time',
        x_label='Circuit Width (qubits)',
        y_label='Optimization Time (seconds)',
        title='Optimization Time Scaling'
    )
    
    # Create performance dashboard
    # Note: This will start a server, so it's commented out for the example
    # dashboard = PerformanceDashboard(port=8050)
    # dashboard.create_dashboard(benchmark_results)