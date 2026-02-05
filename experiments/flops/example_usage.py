"""
Example usage of the modular plotting system.
This demonstrates how to use the 5 plotting functions with CUPTI data collection.
"""

import argparse
import torch
import torch.nn as nn
from plot_profiler import PlotProfiler

# Example model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        return self.layers(x)


def example_usage(show: bool = False):
    """Example of how to use the PlotProfiler API.
    
    Args:
        show: If True, display plots with plt.show(). Otherwise, just generate them.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    X = torch.randn(1, 784).to(device)
    
    # Define the function to profile
    def run_model():
        return model(X)
    
    # Create profiler and run
    profiler = PlotProfiler(run_model)
    result = profiler()
    
    # Generate all plots
    profiler.visualize_all(show=show)
    
    if not show:
        print("All plots generated! Use --show flag to display them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example usage of the modular plotting system")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots (default: False)"
    )
    args = parser.parse_args()
    
    example_usage(show=args.show)
