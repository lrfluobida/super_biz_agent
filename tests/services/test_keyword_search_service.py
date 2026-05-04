import os
import tempfile
import unittest

from app.services.keyword_search_service import BM25Encoder, _sparse_row_to_dict


class BM25EncoderTestCase(unittest.TestCase):
    def setUp(self):
        self.encoder = BM25Encoder()
        self.corpus = [
            "CPU 使用率超过 90% 触发告警，需要检查进程负载",
            "内存泄漏导致 OOM Killer 杀掉主进程，建议监控 RES 指标",
            "磁盘 I/O 延迟升高可能是由于频繁的日志写入操作",
        ]

    def test_fit_and_encode_documents(self):
        self.encoder.fit(self.corpus)
        self.assertTrue(self.encoder.is_fitted)

        vecs = self.encoder.encode_documents(self.corpus)
        self.assertEqual(len(vecs), len(self.corpus))

        for v in vecs:
            self.assertIsInstance(v, dict)
            self.assertGreater(len(v), 0)
            for k, val in v.items():
                self.assertIsInstance(k, int)
                self.assertIsInstance(val, float)

    def test_fit_and_encode_query(self):
        self.encoder.fit(self.corpus)
        qvec = self.encoder.encode_query("CPU 高负载如何处理")
        self.assertIsInstance(qvec, dict)
        self.assertGreater(len(qvec), 0)

    def test_encode_without_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.encoder.encode_documents(["test"])

        with self.assertRaises(RuntimeError):
            self.encoder.encode_query("test")

    def test_empty_corpus_fit(self):
        self.encoder.fit([])
        self.assertFalse(self.encoder.is_fitted)

    def test_empty_encode_documents(self):
        self.encoder.fit(self.corpus)
        vecs = self.encoder.encode_documents([])
        self.assertEqual(vecs, [])

    def test_save_and_load(self):
        self.encoder.fit(self.corpus)
        original_vecs = self.encoder.encode_documents(self.corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bm25_test.pkl")
            self.encoder.save(path)
            self.assertTrue(os.path.exists(path))

            encoder2 = BM25Encoder()
            encoder2.load(path)
            reloaded_vecs = encoder2.encode_documents(self.corpus)

            self.assertEqual(len(original_vecs), len(reloaded_vecs))
            for orig, reloaded in zip(original_vecs, reloaded_vecs):
                self.assertEqual(orig, reloaded)

    def test_sparse_row_to_dict(self):
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import (
            build_default_analyzer,
        )

        analyzer = build_default_analyzer(language="zh")
        bm25 = BM25EmbeddingFunction(analyzer)
        corpus = ["测试文本"]
        bm25.fit(corpus)
        mat = bm25.encode_documents(corpus)

        result = _sparse_row_to_dict(mat, 0)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, float)
