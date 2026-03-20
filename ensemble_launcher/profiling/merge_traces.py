"""Utility to merge multiple Perfetto traces into a single file.

When running distributed ensembles, each master/worker generates its own trace.
This script combines them for unified visualization in Perfetto UI.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List


def merge_perfetto_traces(trace_files: List[str], output_file: str, no_timestamp_merge: bool = False):
    """Merge multiple Perfetto trace files into one.
    
    This function aligns traces from different nodes using their base timestamps.
    Each trace has a base_timestamp_seconds field (from time.perf_counter() on that node).
    We find the earliest base timestamp and adjust all events relative to it.
    
    Args:
        trace_files: List of input trace file paths
        output_file: Output merged trace file path
        no_timestamp_merge: If True, skip timestamp alignment (keep original timestamps)
    """
    all_events = []
    
    # First pass: find the minimum base timestamp across all traces
    trace_info = []
    min_base_timestamp = float('inf')
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
            base_ts = trace_data.get('base_timestamp_seconds', 0)
            min_base_timestamp = min(min_base_timestamp, base_ts)
            trace_info.append({
                'file': trace_file,
                'data': trace_data,
                'base_timestamp': base_ts
            })
    
    print(f"Global base timestamp: {min_base_timestamp} seconds")
    
    # Second pass: adjust all events based on base timestamp differences
    for i, info in enumerate(trace_info):
        trace_file = info['file']
        trace_data = info['data']
        base_ts = info['base_timestamp']
        
        print(f"\nProcessing {trace_file}...")
        print(f"  Base timestamp: {base_ts} seconds")
        
        # Calculate time offset in microseconds
        # Events in this trace are relative to base_ts, we need them relative to min_base_timestamp
        if no_timestamp_merge:
            time_offset_us = 0
            print(f"  Time offset: skipped (no-timestamp-merge enabled)")
        else:
            time_offset_us = int((base_ts - min_base_timestamp) * 1_000_000)
            print(f"  Time offset: {time_offset_us} μs ({time_offset_us / 1_000_000:.6f} seconds) for file {trace_file}")
        
        events = trace_data.get('traceEvents', [])
        
        # Adjust PIDs to be unique per file to avoid conflicts
        pid_offset = i * 10000
        
        for event in events:
            # Adjust PID to make it unique across traces
            if 'pid' in event:
                event['pid'] = event['pid'] + pid_offset
            
            # Adjust timestamp to align with global base timestamp
            # Skip metadata events (M) as they don't have meaningful timestamps
            if 'ts' in event and event.get('ph') != 'M':
                event['ts'] = event['ts'] + time_offset_us
            
            all_events.append(event)
        
        print(f"  Added {len(events)} events")
    
    # Write merged trace
    merged_trace = {
        'traceEvents': all_events,
        'displayTimeUnit': 'ms',
        'base_timestamp_seconds': min_base_timestamp,
    }
    
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f, indent=2)
    
    print(f"\nMerged trace written to: {output_file}")
    print(f"Total events: {len(all_events)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Perfetto trace files into one"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='profiles',
        help='Directory containing perfetto trace files (default: profiles)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='profiles/merged_perfetto.json',
        help='Output merged trace file (default: profiles/merged_perfetto.json)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_perfetto.json',
        help='File pattern to match (default: *_perfetto.json)'
    )
    parser.add_argument(
        '--no-timestamp-merge',
        action='store_true',
        help='Skip timestamp alignment (keep original timestamps from each trace)'
    )
    
    args = parser.parse_args()
    
    # Find all matching trace files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Directory {args.input_dir} does not exist")
        return
    
    trace_files = list(input_path.glob(args.pattern))
    
    # Exclude the output file if it exists in the same directory
    output_path = Path(args.output)
    trace_files = [str(f) for f in trace_files if f != output_path]
    
    if not trace_files:
        print(f"No trace files matching '{args.pattern}' found in {args.input_dir}")
        return
    
    print(f"Found {len(trace_files)} trace files:")
    for f in trace_files:
        print(f"  - {f}")
    print()
    
    merge_perfetto_traces(trace_files, args.output, args.no_timestamp_merge)
    
    print(f"\nTo visualize:")
    print(f"1. Open https://ui.perfetto.dev")
    print(f"2. Click 'Open trace file'")
    print(f"3. Select {args.output}")


if __name__ == '__main__':
    main()
