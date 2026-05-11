"""DriftGuard 单元测试 — 余弦相似度计算和阈值分类"""

import pytest

from app.models.rewrite import DriftSeverity, DriftCheckResult
from app.services.drift_guard import DriftGuard, _cosine_similarity


class TestCosineSimilarity:
    """余弦相似度计算测试"""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_semantically_similar(self):
        a = [0.5, 0.3, 0.4]
        b = [0.5, 0.35, 0.38]
        sim = _cosine_similarity(a, b)
        assert sim > 0.95

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_norm(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestDriftGuardClassify:
    """阈值分类逻辑测试"""

    def setup_method(self):
        self.guard = DriftGuard(enabled=True)

    def test_high_similarity_use_rewritten(self):
        severity, usable, action = self.guard._classify(0.80, "原查询", "改写查询")
        assert severity == DriftSeverity.LOW
        assert usable is True
        assert action == "use_rewritten"

    def test_boundary_at_threshold(self):
        severity, usable, action = self.guard._classify(0.65, "原查询", "改写查询")
        assert severity == DriftSeverity.LOW
        assert usable is True

    def test_moderate_drift_use_both(self):
        severity, usable, action = self.guard._classify(0.50, "原查询", "改写查询")
        assert severity == DriftSeverity.MODERATE
        assert usable is True
        assert action == "use_both"

    def test_boundary_at_moderate_threshold(self):
        severity, usable, action = self.guard._classify(0.40, "原查询", "改写查询")
        assert severity == DriftSeverity.MODERATE
        assert usable is True

    def test_high_severity_drop_rewrite(self):
        severity, usable, action = self.guard._classify(0.20, "原查询", "改写查询")
        assert severity == DriftSeverity.HIGH
        assert usable is False
        assert action == "use_original"


class TestDriftGuardCheck:
    """check() 方法集成测试（禁用状态）"""

    def test_same_query_returns_identical(self):
        guard = DriftGuard(enabled=True)
        # 不调用 embed API（因为没有真实服务），走 same query 逻辑
        import asyncio
        # same query shortcut
        result = asyncio.run(guard.check("test", "test"))
        assert isinstance(result, DriftCheckResult)
        assert result.similarity == 1.0
        assert result.severity == DriftSeverity.NONE

    def test_disabled_guard_skips_check(self):
        guard = DriftGuard(enabled=False)
        import asyncio
        result = asyncio.run(guard.check("原查询", "改写查询"))
        assert result.similarity == 1.0
        assert result.severity == DriftSeverity.NONE
        assert result.action == "use_rewritten"

    def test_empty_queries_shortcut(self):
        guard = DriftGuard(enabled=True)
        import asyncio
        result = asyncio.run(guard.check("", "改写"))
        assert result.similarity == 1.0
