#!/usr/bin/env python3
"""
Plot strong scaling results from benchmark_worker_uniform_runtime.py

This script visualizes the runtime and efficiency of sync and async workers
as task granularity varies while keeping total work constant.
"""

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import argparse
from pathlib import Path
import json

# Use scienceplots style
plt.style.use(['science', 'no-latex'])


def plot_strong_scaling(csv_file, output_dir=None):
    """
    Create plots from strong scaling benchmark results.
    
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
        total_work = metadata.get('total_work_per_core', 10.0)
        ncores = metadata.get('ncores', 'N')
    else:
        total_work = df['total_work_per_core'].iloc[0]
        ncores = df['ncores'].iloc[0]
    
    # Calculate task granularity (seconds per task)
    df['task_granularity'] = df['sleep_time']
    
    print(f"Loaded {len(df)} data points")
    print(f"Total work per core: {total_work}s")
    print(f"Number of cores: {ncores}")
    print(f"Task granularity range: {df['task_granularity'].min():.4f}s - {df['task_granularity'].max():.1f}s")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Strong Scaling Performance (Total Work = {total_work}s/core, {ncores} cores)', 
                 fontsize=14, y=0.995)
    
    # Plot 1: Runtime vs Task Granularity
    ax1 = axes[0, 0]
    ax1.plot(df['task_granularity'], df['sync_runtime'], 'o-', label='Sync', markersize=4)
    ax1.plot(df['task_granularity'], df['async_runtime'], 's-', label='Async', markersize=4)
    ax1.axhline(y=total_work, color='k', linestyle='--', linewidth=1, label='Ideal', alpha=0.5)
    ax1.set_xlabel('Task Granularity (s/task)')
    ax1.set_ylabel('Total Runtime (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Runtime vs Task Granularity')
    
    # Plot 2: Efficiency vs Task Granularity
    ax2 = axes[0, 1]
    ax2.plot(df['task_granularity'], df['sync_efficiency'] * 100, 'o-', label='Sync', markersize=4)
    ax2.plot(df['task_granularity'], df['async_efficiency'] * 100, 's-', label='Async', markersize=4)
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Task Granularity (s/task)')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_xscale('log')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Parallel Efficiency')
    
    # Plot 3: Overhead vs Number of Tasks
    ax3 = axes[1, 0]
    ax3.plot(df['ntasks_per_core'], df['sync_overhead'], 'o-', label='Sync', markersize=4)
    ax3.plot(df['ntasks_per_core'], df['async_overhead'], 's-', label='Async', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Tasks per Core')
    ax3.set_ylabel('Overhead (s)')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Runtime Overhead')
    
    # Plot 4: Runtime vs Number of Tasks
    ax4 = axes[1, 1]
    ax4.plot(df['ntasks_per_core'], df['sync_runtime'], 'o-', label='Sync', markersize=4)
    ax4.plot(df['ntasks_per_core'], df['async_runtime'], 's-', label='Async', markersize=4)
    ax4.axhline(y=total_work, color='k', linestyle='--', linewidth=1, label='Ideal', alpha=0.5)
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
    
    ax.semilogx(df['task_granularity'], df['sync_efficiency'] * 100, 'o-', 
                label='Sync Worker', markersize=6, linewidth=2)
    ax.semilogx(df['task_granularity'], df['async_efficiency'] * 100, 's-', 
                label='Async Worker', markersize=6, linewidth=2)
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Efficiency')
    
    ax.set_xlabel('Task Granularity (s/task)', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title(f'Strong Scaling Efficiency (Total Work = {total_work}s/core, {ncores} cores)', 
                 fontsize=12)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    # Find 90% efficiency points
    sync_90 = df[df['sync_efficiency'] >= 0.9]
    async_90 = df[df['async_efficiency'] >= 0.9]
    
    if not sync_90.empty:
        min_granularity_sync = sync_90['task_granularity'].min()
        ax.annotate(f'Sync ≥90%: >{min_granularity_sync:.3f}s', 
                   xy=(min_granularity_sync, 90), xytext=(10, -20),
                   textcoords='offset points', fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.5))
    
    if not async_90.empty:
        min_granularity_async = async_90['task_granularity'].min()
        ax.annotate(f'Async ≥90%: >{min_granularity_async:.3f}s', 
                   xy=(min_granularity_async, 90), xytext=(10, 10),
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
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\nBest Performance (highest efficiency):")
    best_sync = df.loc[df['sync_efficiency'].idxmax()]
    best_async = df.loc[df['async_efficiency'].idxmax()]
    
    print(f"\nSync Worker:")
    print(f"  Tasks/core: {best_sync['ntasks_per_core']:.0f}")
    print(f"  Granularity: {best_sync['task_granularity']:.4f}s")
    print(f"  Efficiency: {best_sync['sync_efficiency']:.2%}")
    print(f"  Runtime: {best_sync['sync_runtime']:.3f}s")
    
    print(f"\nAsync Worker:")
    print(f"  Tasks/core: {best_async['ntasks_per_core']:.0f}")
    print(f"  Granularity: {best_async['task_granularity']:.4f}s")
    print(f"  Efficiency: {best_async['async_efficiency']:.2%}")
    print(f"  Runtime: {best_async['async_runtime']:.3f}s")
    
    print("\nWorst Performance (lowest efficiency):")
    worst_sync = df.loc[df['sync_efficiency'].idxmin()]
    worst_async = df.loc[df['async_efficiency'].idxmin()]
    
    print(f"\nSync Worker:")
    print(f"  Tasks/core: {worst_sync['ntasks_per_core']:.0f}")
    print(f"  Granularity: {worst_sync['task_granularity']:.4f}s")
    print(f"  Efficiency: {worst_sync['sync_efficiency']:.2%}")
    print(f"  Runtime: {worst_sync['sync_runtime']:.3f}s")
    
    print(f"\nAsync Worker:")
    print(f"  Tasks/core: {worst_async['ntasks_per_core']:.0f}")
    print(f"  Granularity: {worst_async['task_granularity']:.4f}s")
    print(f"  Efficiency: {worst_async['async_efficiency']:.2%}")
    print(f"  Runtime: {worst_async['async_runtime']:.3f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot strong scaling benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_strong_scaling.py strong_scaling_results.csv
  python plot_strong_scaling.py results.csv --output-dir plots/
        """)
    
    parser.add_argument('csv_file', type=str, 
                       help='CSV file with benchmark results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as CSV file)')
    
    args = parser.parse_args()
    
    plot_strong_scaling(args.csv_file, args.output_dir)
