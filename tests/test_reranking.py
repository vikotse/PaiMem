import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np

from paimem.reranking import VllmReranker, build_reranker, reranker_options


class VllmRerankerTests(unittest.TestCase):
    def transport(self, handler):
        real_client = httpx.Client
        return self.enterContext(patch(
            "httpx.Client",
            side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        ))

    def test_batches_remainder_order_auth_and_probability_scale(self):
        requests = []

        def handler(request):
            requests.append(request)
            body = json.loads(request.content)
            self.assertEqual(str(request.url), "http://localhost:8089/score")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(body["model"], "MemReranker-4B")
            self.assertEqual(body["truncate_prompt_tokens"], 1024)
            self.assertEqual(body["queries"], ["query"] * len(body["documents"]))
            return httpx.Response(200, json={"data": [
                {"index": i, "score": int(doc) / 200}
                for i, doc in reversed(list(enumerate(body["documents"])))
            ]})

        client = self.transport(handler)
        reranker = build_reranker(
            "MemReranker-4B", backend="vllm", base_url="http://localhost:8089/",
            batch_size=16, timeout=7, api_key="test-key",
            device="cuda:99", max_length=1024,  # Device remains local-only.
        )
        values = reranker.score("query", [str(i) for i in range(200)])
        self.assertEqual([len(json.loads(r.content)["documents"]) for r in requests], [16] * 12 + [8])
        np.testing.assert_allclose(values, np.arange(200) / 200)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(client.call_args.kwargs["timeout"], 7)
        self.assertFalse(client.call_args.kwargs["follow_redirects"])

    def test_empty_documents_make_no_request(self):
        with patch("httpx.Client") as client:
            scores = VllmReranker("mem", "http://localhost:8089").score("q", [])
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(scores.dtype, np.float32)
        client.assert_not_called()

    def test_no_auth_header_when_key_unset(self):
        def handler(request):
            self.assertNotIn("authorization", request.headers)
            self.assertEqual(json.loads(request.content)["truncate_prompt_tokens"], 512)
            return httpx.Response(200, json={"data": [{"index": 0, "score": 0.5}]})
        self.transport(handler)
        np.testing.assert_equal(VllmReranker("mem", "http://localhost").score("q", ["d"]), [0.5])

    def test_invalid_responses_are_rejected(self):
        invalid = [
            [], {}, {"data": []}, {"data": [None]},
            {"data": [{"score": 0.5}]},
            {"data": [{"index": True, "score": 0.5}]},
            {"data": [{"index": -1, "score": 0.5}]},
            {"data": [{"index": 1, "score": 0.5}]},
            {"data": [{"index": 0, "score": "0.5"}]},
            {"data": [{"index": 0, "score": True}]},
            {"data": [{"index": 0, "score": None}]},
            {"data": [{"index": 0, "score": float("nan")}]},
            {"data": [{"index": 0, "score": float("inf")}]},
            {"data": [{"index": 0, "score": 1e100}]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                VllmReranker._scores(payload, 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            VllmReranker._scores({"data": [{"index": 0, "score": 0.5}] * 2}, 2)

    def test_http_failure_does_not_return_partial_results_or_retry(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(200, json={"data": [
                    {"index": 0, "score": 0.1}, {"index": 1, "score": 0.2},
                ]})
            return httpx.Response(503, text="private-memory-text")

        self.transport(handler)
        with self.assertRaisesRegex(RuntimeError, r"HTTP 503; batch=2") as caught:
            VllmReranker("mem", "http://localhost", batch_size=2).score("q", ["a", "b", "c", "d", "e"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("private-memory-text", str(caught.exception))

    def test_timeouts_and_connection_errors_are_explicit(self):
        for kind in (httpx.ReadTimeout, httpx.ConnectError):
            with self.subTest(kind=kind):
                def handler(request):
                    raise kind("private detail", request=request)
                real_client = httpx.Client
                with patch("httpx.Client", side_effect=lambda **kw: real_client(
                        transport=httpx.MockTransport(handler), **kw)):
                    with self.assertRaisesRegex(RuntimeError, kind.__name__) as caught:
                        VllmReranker("mem", "http://localhost").score("q", ["d"])
                self.assertNotIn("private detail", str(caught.exception))

    def test_bad_json_and_redirect_fail_without_following(self):
        for response in (httpx.Response(200, text="not-json"),
                         httpx.Response(302, headers={"Location": "http://elsewhere/score"})):
            with self.subTest(status=response.status_code):
                handler = Mock(return_value=response)
                real_client = httpx.Client
                with patch("httpx.Client", side_effect=lambda **kw: real_client(
                        transport=httpx.MockTransport(handler), **kw)):
                    with self.assertRaises((ValueError, RuntimeError)):
                        VllmReranker("mem", "http://localhost").score("q", ["d"])
                self.assertEqual(handler.call_count, 1)

    def test_validation_and_default_factory(self):
        for kwargs in (
            {"model_name": ""}, {"batch_size": 0}, {"batch_size": True},
            {"max_length": 0}, {"max_length": -1}, {"max_length": True},
            {"max_length": 1.5}, {"max_length": "bad"},
            {"timeout": 0}, {"timeout": float("inf")}, {"timeout": "bad"},
            {"base_url": ""}, {"base_url": "file:///tmp"},
            {"base_url": "http://localhost/v1"}, {"base_url": "http://u:p@localhost"},
            {"base_url": "http://localhost?token=secret"}, {"base_url": "http://localhost:bad"},
        ):
            settings = dict(model_name="mem", base_url="http://localhost")
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VllmReranker(**settings)
        with patch("paimem.reranking.CrossEncoderReranker") as local:
            self.assertIs(build_reranker("bge", device="cpu", batch_size=4, max_length=512), local.return_value)
            local.assert_called_once_with(model_name="bge", device="cpu", batch_size=4, max_length=512)
            build_reranker("mem", backend="vllm", base_url="http://localhost")
            self.assertEqual(local.call_count, 1)
            with self.assertRaises(ValueError):
                build_reranker("mem", backend="typo")


class RerankerStartupTests(unittest.TestCase):
    def setUp(self):
        from paimem import api
        self.api = api
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {
            "MEMORY_DB_PATH": str(Path(directory) / "memory.db"),
            "MEMORY_RERANKER_BACKEND": "vllm",
            "MEMORY_RERANKER_MODEL": "MemReranker-4B",
            "MEMORY_RERANKER_BASE_URL": "http://localhost:8089",
            "MEMORY_RERANKER_BATCH_SIZE": "2",
            "MEMORY_RESULT_WINDOW": "0",
        }, clear=True))
        self.enterContext(patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True))
        self.local = self.enterContext(patch("paimem.reranking.CrossEncoderReranker"))
        embedder = Mock()
        embedder.encode_documents.side_effect = lambda texts: np.asarray([[1., 0.] for _ in texts], dtype=np.float32)
        embedder.encode_queries.side_effect = embedder.encode_documents.side_effect
        self.embedder = self.enterContext(patch.object(api, "build_embedder", return_value=embedder))
        api.get_store.cache_clear()
        self.addCleanup(api.get_store.cache_clear)

    def test_environment_retrieval_order_and_probability_threshold(self):
        from paimem.memory_store import MemoryMessage
        os.environ["MEMORY_RERANK_SCORE_THRESHOLD"] = "0.5"
        os.environ["MEMORY_RERANKER_MAX_LENGTH"] = "1024"
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["truncate_prompt_tokens"], 1024)
            requests.append(payload)
            values = [{"index": i, "score": next(value for text, value in (
                ("LOW", 0.25), ("MID", 0.5), ("HIGH", 0.75)) if text in doc)}
                for i, doc in enumerate(payload["documents"])]
            return httpx.Response(200, json={"data": list(reversed(values))})

        real_client = httpx.Client
        with patch("httpx.Client", side_effect=lambda **kw: real_client(
                transport=httpx.MockTransport(handler), **kw)):
            store = self.api.get_store()
            self.assertIsInstance(store.reranker, VllmReranker)
            store.add("r", "u", "s", [MemoryMessage("user", text) for text in ("LOW", "MID", "HIGH")])
            hits = store.search("u", "query", 3)
            self.assertEqual([hit.score for hit in hits], [0.75, 0.5])
            self.assertEqual([len(p["documents"]) for p in requests], [2, 1])
            store.update_config(reranker_enabled=False)
            self.assertEqual(len(store.search("u", "query", 3)), 3)
            self.assertEqual(len(requests), 2)
        self.local.assert_not_called()

    def test_overrides_take_precedence_including_invalid_env_numbers(self):
        os.environ.update(MEMORY_RERANKER_TIMEOUT="bad", MEMORY_RERANKER_BATCH_SIZE="bad",
                          MEMORY_RERANKER_MAX_LENGTH="bad")
        self.api.configure_store(
            reranker_backend="vllm", reranker_model="override",
            reranker_base_url="http://localhost:9999", reranker_api_key="test-key",
            reranker_timeout=8, reranker_batch_size=3, reranker_max_length=768,
        )
        reranker = self.api.get_store().reranker
        self.assertEqual((reranker.model_name, reranker.base_url, reranker.timeout, reranker.batch_size),
                         ("override", "http://localhost:9999", 8, 3))
        self.assertEqual(reranker._api_key, "test-key")
        self.assertEqual(reranker.max_length, 768)
        self.local.assert_not_called()

    def test_invalid_remote_configuration_fails_before_loading_models(self):
        for key, value in (
            ("MEMORY_RERANKER_BACKEND", "typo"), ("MEMORY_RERANKER_MODEL", ""),
            ("MEMORY_RERANKER_BASE_URL", ""), ("MEMORY_RERANKER_TIMEOUT", "-1"),
            ("MEMORY_RERANKER_BATCH_SIZE", "0"),
            ("MEMORY_RERANKER_MAX_LENGTH", "0"),
            ("MEMORY_RERANKER_MAX_LENGTH", "-1"),
            ("MEMORY_RERANKER_MAX_LENGTH", "bad"),
        ):
            with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                with self.assertRaises(ValueError):
                    self.api.get_store()
        self.local.assert_not_called()
        self.embedder.assert_not_called()

    def test_switch_to_local_ignores_unused_remote_settings(self):
        os.environ["MEMORY_RERANKER_TIMEOUT"] = "bad"
        self.api.configure_store(reranker_backend="cross-encoder", reranker_model="bge")
        self.assertIs(self.api.get_store().reranker, self.local.return_value)
        self.local.assert_called_once()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(reranker_options(), {"backend": "cross-encoder"})

    def test_role_name_extractor_reuses_main_remote_reranker(self):
        with patch.dict(os.environ, {
            "MEMORY_ROLE_NAME_ENABLED": "true", "MEMORY_ROLE_NAME_MODEL": "llm",
            "MEMORY_ROLE_NAME_API_KEY": "test-key",
        }), patch("openai.OpenAI"):
            reranker = self.api.get_store().reranker
            extractor = self.api._build_role_name_extractor(reranker, "MemReranker-4B", "cpu")
            self.assertIs(extractor.reranker, reranker)
        self.local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
