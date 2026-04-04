# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KCenterGreedy coreset sampling."""

import pytest
import torch

from anomalib.models.components.sampling import KCenterGreedy


class TestKCenterGreedy:
    """Tests for the KCenterGreedy coreset sampling algorithm."""

    @staticmethod
    def test_coreset_size_matches_sampling_ratio() -> None:
        """Sampled coreset should have exactly floor(n * sampling_ratio) rows."""
        n, d = 1000, 64
        sampling_ratio = 0.1
        embedding = torch.randn(n, d)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        assert coreset.shape == (int(n * sampling_ratio), d)

    @staticmethod
    def test_coreset_is_subset_of_embedding() -> None:
        """Every row in the coreset must come from the original embedding."""
        n, d = 200, 32
        embedding = torch.randn(n, d)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        coreset = sampler.sample_coreset()
        # Coreset rows must each be found in the original embedding
        for row in coreset:
            assert any(torch.allclose(row, embedding[i]) for i in range(n)), "Coreset row not found in embedding"

    @staticmethod
    def test_no_duplicate_indices() -> None:
        """Greedy selection should not pick the same index twice."""
        embedding = torch.randn(500, 64)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.2)
        idxs = sampler.select_coreset_idxs()
        assert len(idxs) == len(set(idxs)), "Duplicate indices found in coreset selection"

    @staticmethod
    @pytest.mark.parametrize("sampling_ratio", [0.05, 0.1, 0.5])
    def test_various_sampling_ratios(sampling_ratio: float) -> None:
        """KCenterGreedy should work correctly for different sampling ratios."""
        n, d = 300, 48
        embedding = torch.randn(n, d)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        expected_size = int(n * sampling_ratio)
        assert coreset.shape == (expected_size, d)

    @staticmethod
    def test_select_coreset_idxs_returns_correct_count() -> None:
        """select_coreset_idxs should return a list of the expected length."""
        n, d = 400, 64
        sampling_ratio = 0.1
        embedding = torch.randn(n, d)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        idxs = sampler.select_coreset_idxs()
        assert len(idxs) == int(n * sampling_ratio)

    @staticmethod
    def test_reset_distances() -> None:
        """reset_distances should set min_distances back to None."""
        embedding = torch.randn(100, 32)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        # Populate min_distances by running an update
        sampler.model.fit(embedding)
        sampler.features = sampler.model.transform(embedding)
        sampler.update_distances(cluster_center=0)
        assert sampler.min_distances is not None
        sampler.reset_distances()
        assert sampler.min_distances is None

    @staticmethod
    def test_update_distances_none_center() -> None:
        """update_distances with None should leave min_distances unchanged."""
        embedding = torch.randn(100, 32)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        sampler.min_distances = None
        sampler.update_distances(cluster_center=None)
        assert sampler.min_distances is None

    @staticmethod
    def test_get_new_idx_raises_on_none_distances() -> None:
        """get_new_idx should raise TypeError when min_distances is None."""
        embedding = torch.randn(100, 32)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        sampler.min_distances = None
        with pytest.raises(TypeError, match="must be of type Tensor"):
            sampler.get_new_idx()

    @staticmethod
    def test_importable_from_components() -> None:
        """KCenterGreedy must be importable directly from anomalib.models.components."""
        from anomalib.models.components import KCenterGreedy as KCG  # noqa: PLC0415

        assert KCG is KCenterGreedy
