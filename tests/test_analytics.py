# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analytics import (
    compute_dataset_stats,
    analyze_action_distribution,
    compute_temporal_stats,
    analyze_annotation_coverage,
    compute_tree_complexity,
    extract_actor_statistics,
    generate_analytics_report,
)


@pytest.fixture
def sample_video():
    return {
        'video_uid': 'test_001',
        'metadata': {'duration': 300},
        'nodes': [
            {
                'node_id': 'root', 'parent_id': None, 'level': 0,
                'start': 0.0, 'end': 300.0, 'plm_caption': 'Cooking demo.',
                'llama3_caption': 'Kitchen scene.',
                'gpt': {
                    'summary': {'brief': 'Cooking video.'},
                    'action': {'brief': 'demonstrate cooking', 'actor': 'A chef'},
                },
            },
            {
                'node_id': 'seg_1', 'parent_id': 'root', 'level': 1,
                'start': 0.0, 'end': 100.0, 'plm_caption': 'Prep.',
                'gpt': {'action': {'brief': 'prepare ingredients', 'actor': 'Hands'}},
            },
            {
                'node_id': 'seg_2', 'parent_id': 'root', 'level': 1,
                'start': 100.0, 'end': 200.0, 'plm_caption': 'Mix.',
                'gpt': {'action': {'brief': 'mix ingredients', 'actor': 'Hands'}},
            },
            {
                'node_id': 'seg_3', 'parent_id': 'root', 'level': 1,
                'start': 200.0, 'end': 300.0, 'plm_caption': 'Cook.',
                'gpt': {'action': {'brief': 'cook on stove', 'actor': 'Person'}},
            },
            {
                'node_id': 'seg_1_1', 'parent_id': 'seg_1', 'level': 2,
                'start': 0.0, 'end': 50.0, 'plm_caption': 'Chop.',
                'gpt': {'action': {'brief': 'chop vegetables', 'actor': 'Hands'}},
            },
            {
                'node_id': 'seg_1_2', 'parent_id': 'seg_1', 'level': 2,
                'start': 50.0, 'end': 100.0, 'plm_caption': 'Measure.',
                'llama3_caption': 'Spices.', 'gpt': None,
            },
        ],
    }


@pytest.fixture
def empty_sample():
    return {'video_uid': 'empty', 'metadata': {}, 'nodes': []}


class TestComputeDatasetStats:
    def test_single_sample(self, sample_video):
        stats = compute_dataset_stats([sample_video])
        assert stats['total_videos'] == 1
        assert stats['total_nodes'] == 6
        assert stats['avg_tree_depth'] == 3.0

    def test_empty_samples(self):
        stats = compute_dataset_stats([])
        assert stats['total_videos'] == 0

    def test_empty_nodes(self, empty_sample):
        stats = compute_dataset_stats([empty_sample])
        assert stats['total_nodes'] == 0


class TestAnalyzeActionDistribution:
    def test_basic(self, sample_video):
        result = analyze_action_distribution([sample_video])
        assert result['total_actions'] == 5
        assert result['unique_actions'] == 5

    def test_top_k(self, sample_video):
        result = analyze_action_distribution([sample_video], top_k=2)
        assert len(result['top_actions']) <= 2

    def test_empty(self):
        result = analyze_action_distribution([])
        assert result['total_actions'] == 0

    def test_verbs(self, sample_video):
        result = analyze_action_distribution([sample_video])
        verbs = [v[0] for v in result['top_verbs']]
        assert any(v in verbs for v in ['demonstrate', 'prepare', 'mix', 'cook', 'chop'])


class TestComputeTemporalStats:
    def test_basic(self, sample_video):
        result = compute_temporal_stats([sample_video])
        assert result['total_segments'] == 6
        assert 'min' in result['duration_stats']

    def test_by_level(self, sample_video):
        result = compute_temporal_stats([sample_video])
        assert 'level_0' in result['by_level']
        assert 'level_1' in result['by_level']

    def test_empty(self):
        result = compute_temporal_stats([])
        assert result['total_segments'] == 0


class TestAnalyzeAnnotationCoverage:
    def test_coverage(self, sample_video):
        result = analyze_annotation_coverage([sample_video])
        assert result['total_nodes'] == 6
        assert result['gpt_coverage'] == pytest.approx(83.33, rel=0.1)
        assert result['plm_coverage'] == 100.0

    def test_llama3(self, sample_video):
        result = analyze_annotation_coverage([sample_video])
        assert result['llama3_coverage'] == pytest.approx(33.33, rel=0.1)

    def test_empty(self):
        result = analyze_annotation_coverage([])
        assert result['total_nodes'] == 0


class TestComputeTreeComplexity:
    def test_basic(self, sample_video):
        result = compute_tree_complexity([sample_video])
        assert result['avg_depth'] == 3.0
        assert result['max_depth'] == 3

    def test_empty(self):
        result = compute_tree_complexity([])
        assert result['avg_depth'] == 0.0


class TestExtractActorStatistics:
    def test_counts(self, sample_video):
        result = extract_actor_statistics([sample_video])
        assert result['total_actors'] == 5

    def test_categories(self, sample_video):
        result = extract_actor_statistics([sample_video])
        assert 'hands' in result['actor_categories'] or 'person' in result['actor_categories']

    def test_top_k(self, sample_video):
        result = extract_actor_statistics([sample_video], top_k=2)
        assert len(result['top_actors']) <= 2


class TestGenerateAnalyticsReport:
    def test_markdown(self, sample_video):
        report = generate_analytics_report([sample_video])
        assert '# Action100M' in report
        assert '## Overview' in report

    def test_text(self, sample_video):
        report = generate_analytics_report([sample_video], output_format='text')
        assert 'ANALYTICS REPORT' in report

    def test_sections(self, sample_video):
        report = generate_analytics_report([sample_video])
        assert 'Actions' in report
        assert 'Coverage' in report


class TestIntegration:
    def test_real_json(self):
        import json
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'hySSAAw4t24.json'
        )
        if not os.path.exists(json_path):
            pytest.skip('Sample JSON not found')

        with open(json_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)

        stats = compute_dataset_stats([sample])
        assert stats['total_nodes'] > 0

        report = generate_analytics_report([sample])
        assert len(report) > 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
