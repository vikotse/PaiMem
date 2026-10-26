import builtins
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from paimem.embeddings import SentenceTransformerEmbedder
from paimem.mps_runtime import inference_guard
from paimem.reranking import CrossEncoderReranker


class MPSRuntimeTests(unittest.TestCase):
    def test_cpu_cuda_do_not_import_torch(self):
        original = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("non-MPS inference imported torch")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import):
            for device in ("cpu", "cuda", "cuda:0"):
                with inference_guard(device):
                    pass

    def test_cross_model_waits_until_gpu_synchronization_finishes(self):
        entered_sync = threading.Event()
        release_sync = threading.Event()
        rerank_started = threading.Event()
        prediction = Mock(return_value=[0.5])

        def synchronize():
            entered_sync.set()
            if not release_sync.wait(5):
                raise AssertionError("synchronization was not released")

        embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
        embedder.batch_size = 64
        embedder.model = SimpleNamespace(device="mps", encode=Mock(return_value=[[1., 0.]]))
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = SimpleNamespace(device="mps:0", predict=prediction)
        reranker._lock = threading.RLock()
        reranker.batch_size = 1

        def rerank():
            rerank_started.set()
            return reranker.score("q", ["d"])

        fake_torch = SimpleNamespace(mps=SimpleNamespace(synchronize=Mock(side_effect=synchronize)))
        with patch.dict("sys.modules", {"torch": fake_torch}), ThreadPoolExecutor(2) as pool:
            embedding = pool.submit(embedder.encode_documents, ["d"])
            try:
                self.assertTrue(entered_sync.wait(5))
                ranking = pool.submit(rerank)
                self.assertTrue(rerank_started.wait(5))
                self.assertFalse(ranking.done())
                prediction.assert_not_called()
            finally:
                release_sync.set()
            np.testing.assert_array_equal(embedding.result(5), [[1., 0.]])
            np.testing.assert_array_equal(ranking.result(5), [0.5])
            self.assertEqual(fake_torch.mps.synchronize.call_count, 2)

    def test_exceptions_synchronize_and_release_lock_for_other_threads(self):
        sync = Mock()
        with patch.dict("sys.modules", {"torch": SimpleNamespace(mps=SimpleNamespace(synchronize=sync))}):
            with self.assertRaisesRegex(ValueError, "inference failed"):
                with inference_guard("mps"):
                    raise ValueError("inference failed")
            sync.assert_called_once()
            sync.side_effect = RuntimeError("sync failed")
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                with inference_guard("mps"):
                    pass
            sync.side_effect = None
            completed = threading.Event()

            def next_request():
                with inference_guard("mps"):
                    completed.set()

            thread = threading.Thread(target=next_request, daemon=True)
            thread.start()
            self.assertTrue(completed.wait(5), "lock leaked after exception")
            thread.join(5)


if __name__ == "__main__":
    unittest.main()
