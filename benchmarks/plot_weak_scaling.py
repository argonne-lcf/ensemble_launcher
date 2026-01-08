#!/usr/bin/env python3
"""
Plot weak scaling results from benchmark_worker_uniform_runtime.py

This script visualizes the runtime and efficiency of sync and async workers
as the number of tasks per core varies while keeping task granularity constant.
"""

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import argparse
from pathlib import Path
import json

# Use scienceplots style
plt.style.use(['science', 'no-latex'])


def plot_weak_scaling(csv_file, output_dir=None):
    """
    Create plots from weak scaling benchmark results.
    
    Args:
        csv_file: Path to CSV file with results
        output_dir: Directory to save plots (default: same as CSV file)
    """
    # Load data
    df = pd.read_csv(csv_file)
    csv_path = Path(csv_file)
    
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata if available
    metadata_file = csv_path.with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        sleep_time = metadata.get('sleep_time', 0.1)
        ncores = metadata.get('ncores', 'N')
    else:
        sleep_time = df['sleep_time'].iloc[0]
        ncores = df['ncores'].iloc[0]
    
    print(f"Loaded {len(df)} data points")
    print(f"Task sleep time: {sleep_time}s")
    print(f"Number of cores: {ncores}")
    print(f"Tasks per core range: {df['ntasks_per_core'].min()} - {df['ntasks_per_core'].max()}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Weak Scaling Performance (Task Granularity = {sleep_time}s, {ncores} cores)', 
                 fontsize=14, y=0.995)
    
    # Plot 1: Runtime vs Work per Core
    ax1 = axes[0, 0]
    ax1.plot(df['work_per_core'], df['sync_runtime'], 'o-', label='Sync', markersize=4)
    ax1.plot(df['work_per_core'], df['async_runtime'], 's-', label='Async', markersize=4)
    ax1.plot(df['work_per_core'], df['ideal_runtime'], 'k--', linewidth=1, label='Ideal', alpha=0.5)
    ax1.set_xlabel('Work per Core (s)')
    ax1.set_ylabel('Total Runtime (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Runtime vs Work per Core')
    
    # Plot 2: Efficiency vs Work per Core
    ax2 = axes[0, 1]
    ax2.plot(df['work_per_core'], df['sync_efficiency'] * 100, 'o-', label='Sync', markersize=4)
    ax2.plot(df['work_per_core'], df['async_efficiency'] * 100, 's-', label='Async', markersize=4)
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Work per Core (s)')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_xscale('log')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Parallel Efficiency')
    
    # Plot 3: Overhead vs Work per Core
    ax3 = axes[1, 0]
    ax3.plot(df['work_per_core'], df['sync_overhead'], 'o-', label='Sync', markersize=4)
    ax3.plot(df['work_per_core'], df['async_overhead'], 's-', label='Async', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Work per Core (s)')
    ax3.set_ylabel('Overhead (s)')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Runtime Overhead')
    
    # Plot 4: Runtime vs Number of Tasks per Core
    ax4 = axes[1, 1]
    ax4.plot(df['ntasks_per_core'], df['sync_runtime'], 'o-', label='Sync', markersize=4)
    ax4.plot(df['ntasks_per_core'], df['async_runtime'], 's-', label='Async', markersize=4)
    ax4.plot(df['ntasks_per_core'], df['ideal_runtime'], 'k--', linewidth=1, label='Ideal', alpha=0.5)
    ax4.set_xlabel('Tasks per Core')
    ax4.set_ylabel('Total Runtime (s)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Runtime vs Task Count')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{csv_path.stem}_plots.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file.absolute()}")
    
    # Also save as PNG
    output_file_png = output_dir / f"{csv_path.stem}_plots.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file_png.absolute()}")
    
    plt.show()
    
    # Create additional detailed efficiency plot
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogx(df['ntasks_per_core'], df['sync_efficiency'] * 100, 'o-', 
                label='Sync Worker', markersize=6, linewidth=2)
    ax.semilogx(df['ntasks_per_core'], df['async_efficiency'] * 100, 's-', 
                label='Async Worker', markersize=6, linewidth=2)
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Efficiency')
    
    ax.set_xlabel('Tasks per Core', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title(f'Weak Scaling Efficiency (Task Granularity = {sleep_time}s, {ncores} cores)', 
                 fontsize=12)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    # Find 90% efficiency points
    sync_90 = df[df['sync_efficiency'] >= 0.9]
    async_90 = df[df['async_efficiency'] >= 0.9]
    
    if not sync_90.empty:
        max_tasks_sync = sync_90['ntasks_per_core'].max()
        ax.annotate(f'Sync ≥90%: up to {max_tasks_sync} tasks/core', 
                   xy=(max_tasks_sync, 90), xytext=(10, -20),
                   textcoords='offset points', fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.5))
    
    if not async_90.empty:
        max_tasks_async = async_90['ntasks_per_core'].max()
        ax.annotate(f'Async ≥90%: up to {max_tasks_async} tasks/core', 
                   xy=(max_tasks_async, 90), xytext=(10, 10),
                   textcoords='offset points', fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.5))
    
    plt.tight_layout()
    
    output_file2 = output_dir / f"{csv_path.stem}_efficiency.pdf"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {output_file2.absolute()}")
    
    output_file2_png = output_dir / f"{csv_path.stem}_efficiency.png"
    plt.savefig(output_file2_png, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {output_file2_png.absolute()}")
    
    plt.show()
    
    # Create speedup plot
    fig3, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate speedup relative to smallest configuration
    sync_speedup = df['sync_runtime'].iloc[0] / df['sync_runtime']
    async_speedup = df['async_runtime'].iloc[0] / df['async_runtime']
    ideal_speedup = df['work_per_core'] / df['work_per_core'].iloc[0]
    
    ax.loglog(df['work_per_core'], sync_speedup, 'o-', 
             label='Sync Worker', markersize=6, linewidth=2)
    ax.loglog(df['work_per_core'], async_speedup, 's-', 
             label='Async Worker', markersize=6, linewidth=2)
    ax.loglog(df['work_per_core'], ideal_speedup, 'k--', 
             linewidth=1, label='Linear Speedup', alpha=0.5)
    
    ax.set_xlabel('Work per Core (s)', fontsize=12)
    ax.set_ylabel('Relative Speedup', fontsize=12)
    ax.set_title(f'Weak Scaling Speedup (Task Granularity = {sleep_time}s, {ncores} cores)', 
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file3 = output_dir / f"{csv_path.stem}_speedup.pdf"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to: {output_file3.absolute()}")
    
    output_file3_png = output_dir / f"{csv_path.stem}_speedup.png"
    plt.savefig(output_file3_png, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to: {output_file3_png.absolute()}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\nBest Performance (highest efficiency):")
    best_sync = df.loc[df['sync_efficiency'].idxmax()]
    best_async = df.loc[df['async_efficiency'].idxmax()]
    
    print(f"\nSync Worker:")
    print(f"  Tasks/core: {best_sync['ntasks_per_core']:.0f}")
    print(f"  Work/core: {best_sync['work_per_core']:.2f}s")
    print(f"  Efficiency: {best_sync['sync_efficiency']:.2%}")
    print(f"  Runtime: {best_sync['sync_runtime']:.3f}s")
    
    print(f"\nAsync Worker:")
    print(f"  Tasks/core: {best_async['ntasks_per_core']:.0f}")
    print(f"  Work/core: {best_async['work_per_core']:.2f}s")
    print(f"  Efficiency: {best_async['async_efficiency']:.2%}")
    print(f"  Runtime: {best_async['async_runtime']:.3f}s")
    
    print("\nWorst Performance (lowest efficiency):")
    worst_sync = df.loc[df['sync_efficiency'].idxmin()]
    worst_async = df.loc[df['async_efficiency'].idxmin()]
    
    print(f"\nSync Worker:")
    print(f"  Tasks/core: {worst_sync['ntasks_per_core']:.0f}")
    print(f"  Work/core: {worst_sync['work_per_core']:.2f}s")
    print(f"  Efficiency: {worst_sync['sync_efficiency']:.2%}")
    print(f"  Runtime: {worst_sync['sync_runtime']:.3f}s")
    
    print(f"\nAsync Worker:")
    print(f"  Tasks/core: {worst_async['ntasks_per_core']:.0f}")
    print(f"  Work/core: {worst_async['work_per_core']:.2f}s")
    print(f"  Efficiency: {worst_async['async_efficiency']:.2%}")
    print(f"  Runtime: {worst_async['async_runtime']:.3f}s")
    
    # Print scaling characteristics
    print("\nScaling Characteristics:")
    print("-" * 60)
    
    # Calculate average efficiency
    sync_avg_eff = df['sync_efficiency'].mean()
    async_avg_eff = df['async_efficiency'].mean()
    
    print(f"Average Sync Efficiency: {sync_avg_eff:.2%}")
    print(f"Average Async Efficiency: {async_avg_eff:.2%}")
    
    # Find the range where efficiency stays above 90%
    sync_90_range = df[df['sync_efficiency'] >= 0.9]
    async_90_range = df[df['async_efficiency'] >= 0.9]
    
    if not sync_90_range.empty:
        print(f"\nSync ≥90% efficiency range:")
        print(f"  Tasks/core: {sync_90_range['ntasks_per_core'].min()} - {sync_90_range['ntasks_per_core'].max()}")
        print(f"  Work/core: {sync_90_range['work_per_core'].min():.2f}s - {sync_90_range['work_per_core'].max():.2f}s")
    
    if not async_90_range.empty:
        print(f"\nAsync ≥90% efficiency range:")
        print(f"  Tasks/core: {async_90_range['ntasks_per_core'].min()} - {async_90_range['ntasks_per_core'].max()}")
        print(f"  Work/core: {async_90_range['work_per_core'].min():.2f}s - {async_90_range['work_per_core'].max():.2f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot weak scaling benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_weak_scaling.py weak_scaling_results.csv
  python plot_weak_scaling.py results.csv --output-dir plots/
        """)
    
    parser.add_argument('csv_file', type=str, 
                       help='CSV file with benchmark results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as CSV file)')
    
    args = parser.parse_args()
    
    plot_weak_scaling(args.csv_file, args.output_dir)
