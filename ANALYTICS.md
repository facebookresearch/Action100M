# Analytics Module

Dataset analytics utilities for Action100M.

## Why

Researchers need to understand dataset characteristics before training models. This module provides statistical analysis of action distributions, temporal patterns, and annotation coverage.

## Functions

| Function | Purpose |
|----------|---------|
| `compute_dataset_stats` | Video/node counts, duration, annotation coverage |
| `analyze_action_distribution` | Action frequency, verb extraction, top-k ranking |
| `compute_temporal_stats` | Segment durations by hierarchy level |
| `analyze_annotation_coverage` | GPT vs PLM vs Llama3 coverage comparison |
| `compute_tree_complexity` | Tree depth, branching factor, balance metrics |
| `extract_actor_statistics` | Actor categorization (hands, person, tools, etc) |
| `generate_analytics_report` | Full markdown or text report |

## Usage

```python
from utils.analytics import (
    compute_dataset_stats,
    analyze_action_distribution,
    generate_analytics_report,
)

# Load your samples
samples = [...]  # List of video samples with 'nodes' field

# Get statistics
stats = compute_dataset_stats(samples)
print(f"Videos: {stats['total_videos']}")
print(f"Nodes: {stats['total_nodes']}")
print(f"GPT Coverage: {stats['annotation_coverage']}%")

# Action distribution
actions = analyze_action_distribution(samples, top_k=20)
for action, count in actions['top_actions']:
    print(f"{action}: {count}")

# Full report
report = generate_analytics_report(samples)
print(report)
```

## Output Example

```
Dataset Statistics:
  total_videos: 1
  total_nodes: 917
  total_duration_hours: 0.19
  avg_nodes_per_video: 917.0
  avg_tree_depth: 17.0
  annotation_coverage: 28.57%

Top Actions:
  apply watercolor: 12
  write text: 8
  cut paper: 7
  glue elements: 5
```

## Tests

```bash
python -m pytest tests/test_analytics.py -v
```

23 test cases covering all functions and edge cases.
