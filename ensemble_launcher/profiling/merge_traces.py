"""Utility to merge multiple Perfetto traces into a single file.

When running distributed ensembles, each master/worker generates its own trace.
This script combines them for unified visualization in Perfetto UI.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List


def merge_perfetto_traces(trace_files: List[str], output_file: str):
    """Merge multiple Perfetto trace files into one.
    
    Args:
        trace_files: List of input trace file paths
        output_file: Output merged trace file path
    """
    all_events = []
    
    # Find the earliest timestamp across all traces
    min_timestamp = float('inf')
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
            events = trace_data.get('traceEvents', [])
            
            # Find earliest timestamp in this trace
            for event in events:
                if 'ts' in event and event['ph'] != 'M':  # Ignore metadata events
                    min_timestamp = min(min_timestamp, event['ts'])
    
    print(f"Base timestamp: {min_timestamp} μs")
    
    # Load and adjust all events
    for i, trace_file in enumerate(trace_files):
        print(f"Processing {trace_file}...")
        
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
            events = trace_data.get('traceEvents', [])
            
            # Adjust PIDs to be unique per file to avoid conflicts
            pid_offset = i * 10000
            
            for event in events:
                # Adjust PID to make it unique
                if 'pid' in event:
                    event['pid'] = event['pid'] + pid_offset
                
                all_events.append(event)
            
            print(f"  Added {len(events)} events from {trace_file}")
    
    # Write merged trace
    merged_trace = {
        'traceEvents': all_events,
        'displayTimeUnit': 'ms',
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
    
    merge_perfetto_traces(trace_files, args.output)
    
    print(f"\nTo visualize:")
    print(f"1. Open https://ui.perfetto.dev")
    print(f"2. Click 'Open trace file'")
    print(f"3. Select {args.output}")


if __name__ == '__main__':
    main()
