# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Aesthetic configuration
PLOT_CONFIG = {
    'figure_width': 20,
    'row_height': 1.5,
    'min_height': 6,
    'box_height': 0.7,
    'box_y_offset': 0.35,
    'time_label_offset': 0.45,
    'rotation_threshold': 15,  # seconds
    'text_truncate_long': 80,
    'text_truncate_medium': 50,
    'words_per_line': 2,
    'font_family': 'sans-serif',
    'font_size_main': 8,
    'font_size_time': 7,
    'font_size_xlabel': 12,
    'font_size_ylabel': 12,
    'font_size_title': 14,
    'box_linewidth': 2,
    'box_alpha': 0.8,
    'box_edge_color': 'black',
    'connection_line_style': 'k--',
    'connection_line_alpha': 0.3,
    'connection_line_width': 1,
    'time_start_color': 'darkblue',
    'time_end_color': 'darkred',
    'colormap': plt.cm.Set3,
    'grid_alpha': 0.3,
}

def plot_tree_timeline(nodes, config=None):
    """
    Plot a hierarchical tree visualization of actions with time intervals.
    
    Args:
        nodes: List of node dictionaries with hierarchical action data
        config: Optional dictionary to override default aesthetic settings
    """
    if not nodes:
        print("No nodes to plot")
        return
    
    # Merge custom config with defaults
    cfg = {**PLOT_CONFIG, **(config or {})}
    
    # Build parent-child relationships
    node_dict = {n['node_id']: n for n in nodes}
    children_map = {}
    for node in nodes:
        parent_id = node['parent_id']
        if parent_id not in children_map:
            children_map[parent_id] = []
        children_map[parent_id].append(node['node_id'])
    
    # Find root nodes (no parent)
    roots = [n for n in nodes if n['parent_id'] is None]
    
    # Calculate positions using BFS
    y_positions = {}
    
    def assign_positions(node_id, level):
        """Recursively assign vertical positions based on tree level"""
        if node_id not in y_positions:
            y_positions[node_id] = level
        
        if node_id in children_map:
            for child_id in children_map[node_id]:
                assign_positions(child_id, level + 1)
    
    for root in roots:
        assign_positions(root['node_id'], 0)
    
    # Get max level for plot sizing
    max_level = max(y_positions.values()) if y_positions else 0
    
    # Get time range
    all_times = [(n['start'], n['end']) for n in nodes]
    min_time = min(t[0] for t in all_times)
    max_time = max(t[1] for t in all_times)
    time_range = max_time - min_time
    
    # Create figure
    fig, ax = plt.subplots(figsize=(cfg['figure_width'], 
                                     max(cfg['min_height'], (max_level + 1) * cfg['row_height'])))
    
    # Color palette for different levels
    colors = cfg['colormap'](np.linspace(0, 1, max_level + 2))
    
    # Plot each node as a rectangle with time span
    for node in nodes:
        node_id = node['node_id']
        level = y_positions[node_id]
        start = node['start']
        end = node['end']
        duration = end - start
        
        # Get action text
        gpt_data = node.get('gpt') or {}
        action_data = gpt_data.get('action') or {}
        action_text = action_data.get('brief', 'No action')
        
        # Skip nodes with no action or N/A
        if not action_text or action_text in ['No action', 'N/A', 'n/a', 'NA']:
            continue
        
        # Format text based on duration
        rotation = 0
        display_text = action_text
        
        # Truncate if too long
        if len(action_text) > cfg['text_truncate_long']:
            display_text = action_text[:cfg['text_truncate_long']-3] + "..."
        
        if duration < cfg['rotation_threshold']:
            # For short/medium time intervals, rotate text vertically 
            rotation = 90
            words = display_text.split()
            if len(words) > cfg['words_per_line']:
                # Break every N words
                lines = []
                for i in range(0, len(words), cfg['words_per_line']):
                    lines.append(' '.join(words[i:i+cfg['words_per_line']]))
                display_text = '\n'.join(lines)
        else:
            # For long intervals, just truncate if needed
            if len(action_text) > cfg['text_truncate_medium']:
                display_text = action_text[:cfg['text_truncate_medium']-3] + "..."
        
        # Position: x based on time, y based on level
        y = -level  # Invert so root is at top
        
        # Draw rectangle for time span
        rect = FancyBboxPatch(
            (start, y - cfg['box_y_offset']), duration, cfg['box_height'],
            boxstyle="round,pad=0.05",
            edgecolor=cfg['box_edge_color'],
            facecolor=colors[level],
            linewidth=cfg['box_linewidth'],
            alpha=cfg['box_alpha']
        )
        ax.add_patch(rect)
        
        # Add text label
        text_x = start + duration / 2
        text_y = y
        ax.text(text_x, text_y, display_text, 
                ha='center', va='center',
                fontsize=cfg['font_size_main'], 
                family=cfg['font_family'],
                rotation=rotation,
                wrap=True)
        
        # Add time annotations
        ax.text(start, y - cfg['time_label_offset'], f"{start:.1f}s", 
                ha='center', va='top', 
                fontsize=cfg['font_size_time'], 
                family=cfg['font_family'], 
                color=cfg['time_start_color'])
        ax.text(end, y - cfg['time_label_offset'], f"{end:.1f}s", 
                ha='center', va='top', 
                fontsize=cfg['font_size_time'], 
                family=cfg['font_family'], 
                color=cfg['time_end_color'])
        
        # Draw connection to parent
        if node['parent_id'] is not None and node['parent_id'] in node_dict:
            parent = node_dict[node['parent_id']]
            parent_level = y_positions[node['parent_id']]
            parent_y = -parent_level
            parent_center = (parent['start'] + parent['end']) / 2
            child_center = (start + end) / 2
            
            # Draw line from parent to child
            ax.plot([parent_center, child_center], 
                   [parent_y - cfg['box_y_offset'], y + cfg['box_y_offset']],
                   cfg['connection_line_style'], 
                   alpha=cfg['connection_line_alpha'], 
                   linewidth=cfg['connection_line_width'])
    
    # Styling
    ax.set_xlim(min_time - time_range * 0.05, max_time + time_range * 0.05)
    ax.set_ylim(-max_level - 0.8, 0.8)
    ax.set_xlabel('Time (seconds)', fontsize=cfg['font_size_xlabel'], family=cfg['font_family'])
    ax.set_ylabel('Hierarchy Level', fontsize=cfg['font_size_ylabel'], family=cfg['font_family'])
    
    # Y-axis labels
    ax.set_yticks(range(0, -max_level - 1, -1))
    ax.set_yticklabels([f'Level {i}' for i in range(max_level + 1)])
    
    ax.grid(True, alpha=cfg['grid_alpha'], axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig
