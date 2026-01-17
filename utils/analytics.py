# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List
from collections import Counter, defaultdict
import re


def compute_dataset_stats(samples):
    """Compute statistics across the dataset."""
    if not samples:
        return {
            'total_videos': 0, 'total_nodes': 0, 'total_duration_hours': 0.0,
            'avg_nodes_per_video': 0.0, 'avg_tree_depth': 0.0, 'annotation_coverage': 0.0,
        }

    total_nodes = 0
    total_duration = 0.0
    total_depth = 0
    nodes_with_gpt = 0

    for sample in samples:
        nodes = _get_nodes(sample)
        metadata = sample.get('metadata', {})
        if isinstance(metadata, list) and metadata:
            metadata = metadata[0]

        total_nodes += len(nodes)
        duration = metadata.get('duration', 0)
        if not duration and nodes:
            duration = max(n.get('end', 0) for n in nodes)
        total_duration += duration

        if nodes:
            max_level = max(n.get('level', 0) for n in nodes)
            total_depth += max_level + 1

        for node in nodes:
            if node.get('gpt') is not None:
                nodes_with_gpt += 1

    num_videos = len(samples)
    return {
        'total_videos': num_videos,
        'total_nodes': total_nodes,
        'total_duration_hours': round(total_duration / 3600, 2),
        'avg_nodes_per_video': round(total_nodes / num_videos, 2),
        'avg_tree_depth': round(total_depth / num_videos, 2),
        'annotation_coverage': round(
            nodes_with_gpt / total_nodes * 100 if total_nodes > 0 else 0, 2
        ),
    }


def analyze_action_distribution(samples, field='brief', top_k=50):
    """Analyze action label distribution across the dataset."""
    action_counts = Counter()
    verb_counts = Counter()
    total_actions = 0

    for sample in samples:
        nodes = _get_nodes(sample)
        for node in nodes:
            gpt = node.get('gpt')
            if gpt is None:
                continue

            action_data = gpt.get('action', {})
            action_text = action_data.get(field, '')

            if not action_text or action_text.lower() in ['n/a', 'na', 'none', '']:
                continue

            action_normalized = action_text.strip().lower()
            action_counts[action_normalized] += 1
            total_actions += 1

            words = action_normalized.split()
            if words:
                verb = words[0]
                if verb in ['to', 'a', 'the', 'an']:
                    verb = words[1] if len(words) > 1 else verb
                verb_counts[verb] += 1

    num_videos = len(samples) if samples else 1
    return {
        'total_actions': total_actions,
        'unique_actions': len(action_counts),
        'top_actions': action_counts.most_common(top_k),
        'top_verbs': verb_counts.most_common(20),
        'actions_per_video': round(total_actions / num_videos, 2),
    }


def compute_temporal_stats(samples):
    """Compute temporal statistics for video segments."""
    all_durations = []
    durations_by_level = defaultdict(list)

    for sample in samples:
        nodes = _get_nodes(sample)
        for node in nodes:
            start = node.get('start', 0)
            end = node.get('end', 0)
            duration = end - start
            level = node.get('level', 0)

            if duration > 0:
                all_durations.append(duration)
                durations_by_level[level].append(duration)

    if not all_durations:
        return {
            'total_segments': 0, 'duration_stats': {},
            'by_level': {}, 'segment_counts_by_level': {},
        }

    overall_stats = _compute_duration_stats(all_durations)

    by_level = {}
    segment_counts = {}
    for level in sorted(durations_by_level.keys()):
        by_level[f'level_{level}'] = _compute_duration_stats(durations_by_level[level])
        segment_counts[f'level_{level}'] = len(durations_by_level[level])

    return {
        'total_segments': len(all_durations),
        'duration_stats': overall_stats,
        'by_level': by_level,
        'segment_counts_by_level': segment_counts,
    }


def analyze_annotation_coverage(samples):
    """Analyze annotation source coverage across the dataset."""
    total_nodes = 0
    has_gpt = 0
    has_plm = 0
    has_llama3 = 0
    has_all = 0
    has_none = 0

    for sample in samples:
        nodes = _get_nodes(sample)
        for node in nodes:
            total_nodes += 1

            gpt = node.get('gpt') is not None
            plm = bool(node.get('plm_caption'))
            llama3 = bool(node.get('llama3_caption'))

            if gpt:
                has_gpt += 1
            if plm:
                has_plm += 1
            if llama3:
                has_llama3 += 1
            if gpt and plm and llama3:
                has_all += 1
            if not gpt and not plm and not llama3:
                has_none += 1

    if total_nodes == 0:
        return {
            'total_nodes': 0, 'gpt_coverage': 0.0, 'plm_coverage': 0.0,
            'llama3_coverage': 0.0, 'full_coverage': 0.0, 'no_annotation': 0.0,
        }

    return {
        'total_nodes': total_nodes,
        'gpt_coverage': round(has_gpt / total_nodes * 100, 2),
        'plm_coverage': round(has_plm / total_nodes * 100, 2),
        'llama3_coverage': round(has_llama3 / total_nodes * 100, 2),
        'full_coverage': round(has_all / total_nodes * 100, 2),
        'no_annotation': round(has_none / total_nodes * 100, 2),
    }


def compute_tree_complexity(samples):
    """Analyze hierarchical tree complexity metrics."""
    depths = []
    branching_factors = []
    leaf_counts = []

    for sample in samples:
        nodes = _get_nodes(sample)
        if not nodes:
            continue

        children_map = defaultdict(list)
        node_levels = {}

        for node in nodes:
            node_id = node.get('node_id', '')
            parent_id = node.get('parent_id')
            level = node.get('level', 0)

            node_levels[node_id] = level
            if parent_id:
                children_map[parent_id].append(node_id)

        max_level = max(node_levels.values()) if node_levels else 0
        depths.append(max_level + 1)

        non_leaf_children = [len(c) for c in children_map.values() if c]
        if non_leaf_children:
            branching_factors.append(sum(non_leaf_children) / len(non_leaf_children))

        all_node_ids = set(node_levels.keys())
        parents = set(children_map.keys())
        leaves = all_node_ids - parents
        leaf_counts.append(len(leaves))

    if not depths:
        return {
            'avg_depth': 0.0, 'max_depth': 0, 'avg_branching_factor': 0.0,
            'avg_leaf_count': 0.0, 'balance_score': 0.0,
        }

    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0

    return {
        'avg_depth': round(sum(depths) / len(depths), 2),
        'max_depth': max(depths),
        'avg_branching_factor': round(avg_branching, 2),
        'avg_leaf_count': round(sum(leaf_counts) / len(leaf_counts), 2),
        'balance_score': round(_compute_balance_score(depths, leaf_counts), 2),
    }


def extract_actor_statistics(samples, top_k=30):
    """Extract and analyze actor information from annotations."""
    actor_counts = Counter()
    category_counts = Counter()

    categories = {
        'hands': r'\b(hands?|fingers?)\b',
        'person': r'\b(person|man|woman|individual|someone|chef|worker)\b',
        'group': r'\b(people|group|team|crew)\b',
        'body_parts': r'\b(arm|leg|foot|feet|head|body)\b',
        'tools': r'\b(tool|machine|device|equipment)\b',
    }

    for sample in samples:
        nodes = _get_nodes(sample)
        for node in nodes:
            gpt = node.get('gpt')
            if gpt is None:
                continue

            action_data = gpt.get('action', {})
            actor = action_data.get('actor', '')

            if not actor or actor.lower() in ['n/a', 'na', 'none', '']:
                continue

            actor_normalized = actor.strip().lower()
            actor_counts[actor_normalized] += 1

            categorized = False
            for cat_name, pattern in categories.items():
                if re.search(pattern, actor_normalized, re.IGNORECASE):
                    category_counts[cat_name] += 1
                    categorized = True
                    break
            if not categorized:
                category_counts['other'] += 1

    return {
        'total_actors': sum(actor_counts.values()),
        'unique_actors': len(actor_counts),
        'top_actors': actor_counts.most_common(top_k),
        'actor_categories': dict(category_counts),
    }


def generate_analytics_report(samples, output_format='markdown'):
    """Generate a comprehensive analytics report for the dataset."""
    basic = compute_dataset_stats(samples)
    action = analyze_action_distribution(samples)
    temporal = compute_temporal_stats(samples)
    coverage = analyze_annotation_coverage(samples)
    tree = compute_tree_complexity(samples)
    actor = extract_actor_statistics(samples)

    if output_format == 'markdown':
        return _format_markdown_report(basic, action, temporal, coverage, tree, actor)
    return _format_text_report(basic, action, temporal, coverage, tree, actor)


def _get_nodes(sample):
    """Extract nodes list from sample."""
    nodes = sample.get('nodes', [])
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], list):
        nodes = nodes[0]
    return nodes if isinstance(nodes, list) else []


def _compute_duration_stats(durations):
    """Compute statistical summary for durations."""
    if not durations:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}

    sorted_d = sorted(durations)
    n = len(sorted_d)
    mean = sum(sorted_d) / n
    median = sorted_d[n // 2] if n % 2 else (sorted_d[n//2 - 1] + sorted_d[n//2]) / 2
    variance = sum((x - mean) ** 2 for x in sorted_d) / n
    std = variance ** 0.5

    return {
        'min': round(sorted_d[0], 2),
        'max': round(sorted_d[-1], 2),
        'mean': round(mean, 2),
        'median': round(median, 2),
        'std': round(std, 2),
    }


def _compute_balance_score(depths, leaf_counts):
    """Compute tree balance score (1.0 = perfectly balanced)."""
    if not depths or not leaf_counts:
        return 0.0

    scores = []
    for depth, leaves in zip(depths, leaf_counts):
        ideal_leaves = 2 ** (depth - 1)
        score = min(leaves / ideal_leaves, 1.0) if ideal_leaves > 0 else 0.0
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def _format_markdown_report(basic, action, temporal, coverage, tree, actor):
    """Format analytics as Markdown report."""
    lines = [
        '# Action100M Dataset Analytics Report\n',
        '## Overview\n',
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Videos | {basic['total_videos']} |",
        f"| Total Nodes | {basic['total_nodes']} |",
        f"| Total Duration | {basic['total_duration_hours']} hours |",
        f"| Avg Nodes/Video | {basic['avg_nodes_per_video']} |",
        f"| Annotation Coverage | {basic['annotation_coverage']}% |",
        '',
        '## Actions\n',
        f"- Total: {action['total_actions']}",
        f"- Unique: {action['unique_actions']}",
        '',
        '### Top Actions',
        '| Action | Count |',
        '|--------|-------|',
    ]

    for act, count in action['top_actions'][:10]:
        lines.append(f"| {act[:50]} | {count} |")

    lines.extend([
        '',
        '## Temporal\n',
        f"- Segments: {temporal['total_segments']}",
    ])

    if temporal['duration_stats']:
        d = temporal['duration_stats']
        lines.append(f"- Duration: {d['min']}s - {d['max']}s (mean: {d['mean']}s)")

    lines.extend([
        '',
        '## Coverage\n',
        f"| Source | % |",
        f"|--------|---|",
        f"| GPT | {coverage['gpt_coverage']} |",
        f"| PLM | {coverage['plm_coverage']} |",
        f"| Llama3 | {coverage['llama3_coverage']} |",
        '',
        '## Tree Complexity\n',
        f"- Depth: {tree['avg_depth']} (max: {tree['max_depth']})",
        f"- Branching: {tree['avg_branching_factor']}",
        f"- Leaves: {tree['avg_leaf_count']}",
    ])

    return '\n'.join(lines)


def _format_text_report(basic, action, temporal, coverage, tree, actor):
    """Format analytics as plain text."""
    lines = [
        '=' * 50,
        'ACTION100M ANALYTICS REPORT',
        '=' * 50,
        '',
        f"Videos: {basic['total_videos']}",
        f"Nodes: {basic['total_nodes']}",
        f"Duration: {basic['total_duration_hours']} hours",
        f"Coverage: {basic['annotation_coverage']}%",
        '',
        f"Actions: {action['total_actions']} ({action['unique_actions']} unique)",
        f"Segments: {temporal['total_segments']}",
        f"Tree Depth: {tree['avg_depth']}",
    ]
    return '\n'.join(lines)
