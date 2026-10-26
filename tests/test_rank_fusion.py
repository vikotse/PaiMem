"""Contracts for optional post-MP fusion and runtime configuration."""
import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np
from paimem.rank_fusion import RankFusionConfig, fuse
from paimem.retrieval import retrieve_chunks
from paimem.memory_store import SQLiteMemoryStore, MemoryMessage
from paimem.mempursuit import MemPursuitConfig
from test_mempursuit import CountingEmbedder, VECTORS


class RankFusionTests(unittest.TestCase):
    def retrieve(self, config=None, diagnostics=True, **kwargs):
        rows = [dict(id=str(i), session_id='s', role='user', content=f'MEMORY_{i}',
                     dimension=2, embedding=v.tobytes()) for i,v in enumerate(VECTORS)]
        reranker = Mock(score=lambda q, texts: np.asarray([int(t[-1])+1 for t in texts]))
        settings = dict(embedder=CountingEmbedder(), embedding_lock=threading.RLock(),
                        render_content=lambda row,n:row['content'], retrieval_mode='dense',
                        reranker=reranker, rerank_candidates=4,
                        mempursuit_config=MemPursuitConfig(enabled=True),
                        rank_fusion_config=config, diagnostics=diagnostics)
        settings.update(kwargs)
        return retrieve_chunks(rows, {}, 'question', 2, **settings)

    def test_invalid_parameters(self):
        for args in ({'enabled':1}, {'dense_weight':True}, {'dense_weight':-1},
                     {'rerank_weight':float('nan')}, {'rerank_weight':float('inf')},
                     {'dense_weight':0,'rerank_weight':0}, {'mp_limit':True},
                     {'remaining_limit':-2}, {'mp_limit':1.5}):
            with self.subTest(args=args), self.assertRaises(ValueError): RankFusionConfig(**args)

    def test_capped_prefixes_and_original_fallback_order(self):
        order, detail=fuse([2,0],[3,1],[0,1,2,3],[3,2,1,0],4,
                          RankFusionConfig(enabled=True, dense_weight=0,rerank_weight=1,mp_limit=1,remaining_limit=1))
        self.assertEqual(order.tolist(),[3,2,0,1])
        self.assertEqual(detail['weighted_count'],2)
        self.assertAlmostEqual(detail['rerank'][2],2/3)
        self.assertEqual(len(set(order)),4)

    def test_zero_and_unlimited_and_ties(self):
        for mp, tail, expected in [(0,0,[2,0,3,1]),(-1,-1,[2,0,3,1]),(0,1,[3,2,0,1])]:
            order,_=fuse([2,0],[3,1],[0,1,2,3],[3,2,1,0],4,
                         RankFusionConfig(dense_weight=1,rerank_weight=1,mp_limit=mp,remaining_limit=tail))
            self.assertEqual(order.tolist(),expected)

    def test_matches_rsi_reference_for_random_rankings(self):
        # Compare the implementation used by the original experiment, not copied logic.
        import sys
        with patch.object(sys,'path',[str(Path(__file__).resolve().parents[1]/'scripts'),*sys.path]):
            from reference_rank_fusion import reorder
        rng=np.random.default_rng(48)
        for size in (1,37,230,400):
            dense=rng.permutation(size); ce=rng.permutation(size)
            selected=ce[rng.permutation(size)[:min(100,size)]]
            tail=ce[~np.isin(ce,selected)]; assembly=np.concatenate((selected,tail))
            expected=reorder(assembly,{'dense':dense,'ce':ce},[35,0,0,65,0],size)
            actual,_=fuse(selected,tail,dense,ce,size,RankFusionConfig(enabled=True))
            np.testing.assert_array_equal(actual,expected)

    def test_complete_context_pipeline_and_packed_groups_match_rsi(self):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]/"scripts"), *sys.path]):
            from reference_rank_fusion import build_orders, RetrievalState, make_policy, run_policy
        rng = np.random.default_rng(51)
        vectors = rng.normal(size=(230, 12)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query = rng.normal(size=12).astype(np.float32)
        query /= np.linalg.norm(query)
        docs = [f"D_{i}" for i in range(len(vectors))]
        rows = [dict(id=str(i), session_id='s', role='user', content=docs[i],
                     dimension=12, embedding=v.tobytes()) for i, v in enumerate(vectors)]
        contexts = ["\n".join(docs[max(0,i-1):i+2]) for i in range(len(docs))]
        def encode(texts):
            return np.asarray([vectors[[int(part[2:]) for part in text.splitlines()]].mean(axis=0)
                               for text in texts], dtype=np.float32)
        def score(q, texts):
            return np.asarray([sum(ord(char) for char in text) % 137 for text in texts], dtype=np.float32)
        score_fn = lambda pool: score('D_7', [contexts[i] for i in pool])
        score_fn.query_text = 'D_7'
        orders, _, _ = build_orders(rows, docs, vectors, query, score_fn)
        state = RetrievalState(rows, docs, query, encode(contexts), orders)
        expected_groups, expected_trace = run_policy(state, make_policy('assembly', [20,20,20,20,20]))
        actual = retrieve_chunks(rows, {}, 'D_7', 100,
            embedder=Mock(encode_queries=lambda texts: np.asarray([query]), encode_documents=encode),
            embedding_lock=threading.RLock(), render_content=lambda row,n:row['content'],
            retrieval_mode='hybrid', reranker=Mock(score=score), rerank_candidates=400,
            rerank_small2big_enabled=True, mempursuit_small2big_enabled=True,
            mempursuit_config=MemPursuitConfig(enabled=True,candidates=150),
            rank_fusion_config=RankFusionConfig(enabled=True,dense_weight=20,rerank_weight=20,
                bm25_weight=20,rrf_weight=20,mp_weight=20))
        self.assertEqual(actual.positions.tolist(), expected_trace['assembly_order'])
        store = SQLiteMemoryStore.__new__(SQLiteMemoryStore)
        store.result_window = 1
        store.result_window_seed_k = 20
        groups = store._pack_result_positions(actual.positions, rows, 100)
        self.assertEqual(groups, expected_groups)
        self.assertEqual(len(groups),100)

    def test_five_channels_match_rsi_with_missing_ranks_and_caps(self):
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1]/'scripts'), *sys.path]):
            from reference_rank_fusion import reorder
        rng = np.random.default_rng(82)
        size = 419
        orders = dict(dense=rng.permutation(size), bm25=rng.permutation(size)[:73],
                      rrf=rng.permutation(size), ce=rng.permutation(size)[:400])
        selected = orders['ce'][:100][::-1]
        orders['mp'] = selected
        remaining = orders['ce'][100:]
        assembly = np.concatenate((selected, remaining))
        for weights in ([35,0,0,65,0], [0,0,20,60,20], [20]*5,
                        [0,100,0,0,0], [0,0,100,0,0], [0,0,0,0,100]):
            for mp_limit, remaining_limit in ((-1,-1),(17,29),(0,25)):
                with self.subTest(weights=weights, limits=(mp_limit, remaining_limit)):
                    cfg = RankFusionConfig(enabled=True, dense_weight=weights[0],
                        bm25_weight=weights[1], rrf_weight=weights[2],
                        rerank_weight=weights[3], mp_weight=weights[4],
                        mp_limit=mp_limit, remaining_limit=remaining_limit)
                    candidates = np.concatenate((selected if mp_limit == -1 else selected[:mp_limit],
                        remaining if remaining_limit == -1 else remaining[:remaining_limit]))
                    expected = np.concatenate((reorder(candidates,orders,weights,size),
                                              assembly[~np.isin(assembly,candidates)]))
                    actual, detail = fuse(selected,remaining,orders['dense'],orders['ce'],size,cfg,
                                          bm25_order=orders['bm25'],rrf_order=orders['rrf'])
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(detail['mp'][int(remaining[0])],0)
                    self.assertAlmostEqual(detail['mp'][int(selected[16])],1-16/99)

    def test_new_weights_validation_and_runtime_configuration(self):
        from paimem import api
        from paimem.cli import parse_setting
        store = self.make_store()
        for name in ('bm25','rrf','mp'):
            key = 'rank_fusion_' + name + '_weight'
            for value in (-1, float('nan'),float('inf'),True):
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    RankFusionConfig(**{name+'_weight':value})
            with patch.dict(os.environ, {'MEMORY_'+key.upper():'0.2'},clear=True), patch.dict(api.STORE_INIT_OVERRIDES,{},clear=True):
                self.assertEqual(getattr(api._build_rank_fusion_config(),name+'_weight'),.2)
            self.assertEqual(parse_setting(key+'=0.2'),(key,.2))
            patch_value = api.ConfigPatch(**{key:.2}).model_dump(exclude_none=True)
            store.update_config(**patch_value)
            self.assertEqual(store.get_config()[key],.2)
        store.update_config(rank_fusion_dense_weight=0,rank_fusion_rerank_weight=0)
        before = store.get_config()
        with self.assertRaises(ValueError):
            store.update_config(rank_fusion_bm25_weight=0,rank_fusion_rrf_weight=0,rank_fusion_mp_weight=0)
        self.assertEqual(store.get_config(),before)

    def test_mp_only_pipeline_and_absent_bm25_rrf(self):
        cfg = RankFusionConfig(enabled=True,dense_weight=0,rerank_weight=0,mp_weight=1)
        result = self.retrieve(cfg)
        np.testing.assert_array_equal(result.positions,self.retrieve().positions)
        for row in result.diagnostics['candidates']:
            part = row['channels']['rank_fusion']
            self.assertEqual(part['bm25_rank_score'],0)
            self.assertEqual(part['rrf_rank_score'],0)

    def test_disabled_bypasses_fusion_exactly(self):
        expected=self.retrieve()
        with patch('paimem.retrieval.fuse',side_effect=AssertionError('disabled')):
            actual=self.retrieve(RankFusionConfig())
        np.testing.assert_array_equal(actual.positions,expected.positions)
        np.testing.assert_array_equal(actual.scores,expected.scores)
        self.assertEqual(actual.diagnostics,expected.diagnostics)

    def test_pipeline_and_diagnostics_use_same_order_preserve_scores(self):
        cfg=RankFusionConfig(enabled=True,dense_weight=0,rerank_weight=1)
        result=self.retrieve(cfg); plain=self.retrieve(cfg,diagnostics=False)
        self.assertEqual(result.positions.tolist(),[3,2,1,0])
        np.testing.assert_array_equal(result.positions,plain.positions)
        np.testing.assert_array_equal(result.scores,self.retrieve().scores)
        self.assertTrue(result.diagnostics['rank_fusion']['active'])
        self.assertEqual(result.diagnostics['rank_fusion']['weighted_count'],4)

    def test_threshold_rejections_stay_excluded_full_rank_normalization(self):
        result=self.retrieve(RankFusionConfig(enabled=True),rerank_score_threshold=2)
        self.assertNotIn(0,result.positions)
        row=next(r for r in result.diagnostics['candidates'] if r['id']=='1')
        self.assertAlmostEqual(row['channels']['rank_fusion']['rerank_rank_score'],1/3)

    def test_missing_dependencies_report_reason(self):
        for kwargs,reason in [({'reranker':None},'reranker_disabled'),
                              ({'mempursuit_config':MemPursuitConfig()},'mempursuit_disabled')]:
            result=self.retrieve(RankFusionConfig(enabled=True),**kwargs)
            self.assertFalse(result.diagnostics['rank_fusion']['active'])
            self.assertEqual(result.diagnostics['rank_fusion']['reason'],reason)

    def make_store(self, **kwargs):
        directory=tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        store=SQLiteMemoryStore(Path(directory.name)/'db',CountingEmbedder(),
             retrieval_mode='dense', result_window=0,rerank_candidates=4,
             reranker=Mock(score=lambda q,texts:np.asarray([int(t[-1])+1 for t in texts])),
             mempursuit_config=MemPursuitConfig(enabled=True),**kwargs)
        store.add('r','u','s',[MemoryMessage('user',f'MEMORY_{i}',1704067200000+i*1000) for i in range(4)])
        return store

    def test_runtime_update_atomic_and_toggle_reversible(self):
        store=self.make_store(); before=store.search('u','q',2)
        saved=store.get_config()
        with self.assertRaises(ValueError):
            store.update_config(rank_fusion_enabled=True,rank_fusion_mp_limit=-2,result_window=2)
        self.assertEqual(store.get_config(),saved)
        store.update_config(rank_fusion_enabled=True,rank_fusion_dense_weight=0,rank_fusion_rerank_weight=1)
        self.assertTrue(store.search('u','q',2)[0].content.endswith('MEMORY_3'))
        store.update_config(rank_fusion_enabled=False)
        self.assertEqual(store.search('u','q',2),before)

    def test_conflict_routes_share_snapshot(self):
        from paimem.conflict_strategy import JudgmentQueries
        strategy=Mock(classify=Mock(return_value=JudgmentQueries('p','n','s')),assess=Mock(return_value=None))
        store=self.make_store(conflict_strategy=strategy,rank_fusion_config=RankFusionConfig(enabled=True))
        encode=store.embedder.encode_queries
        def update(texts):
            store.update_config(rank_fusion_enabled=False)
            return encode(texts)
        with patch.object(store.embedder,'encode_queries',side_effect=update):
            _,detail=store.search('u','q',5,diagnostics=True)
        self.assertTrue(detail['rank_fusion']['active'])
        for queue in detail['conflict_strategy']['queues'].values():
            self.assertTrue(queue['pipeline']['rank_fusion']['active'])
        self.assertFalse(store.rank_fusion_config.enabled)

    def test_api_startup_cli_and_runtime(self):
        from paimem import api
        from paimem.cli import parse_setting
        from fastapi.testclient import TestClient
        with patch.dict(os.environ,{'MEMORY_RANK_FUSION_ENABLED':'true','MEMORY_RANK_FUSION_MP_LIMIT':'-1'},clear=True),patch.dict(api.STORE_INIT_OVERRIDES,{},clear=True):
            self.assertTrue(api._build_rank_fusion_config().enabled)
            self.assertEqual(api._build_rank_fusion_config().mp_limit,-1)
            api.STORE_INIT_OVERRIDES['rank_fusion_mp_limit']=10
            self.assertEqual(api._build_rank_fusion_config().mp_limit,10)
            os.environ['MEMORY_RANK_FUSION_DENSE_WEIGHT']='nan'
            with self.assertRaises(ValueError):api._build_rank_fusion_config()
        self.assertEqual(parse_setting('rank_fusion_enabled=true'),('rank_fusion_enabled',True))
        self.assertEqual(parse_setting('rank_fusion_remaining_limit=-1'),('rank_fusion_remaining_limit',-1))
        store=self.make_store()
        with patch.dict(api.app.dependency_overrides,{api.get_store:lambda:store}),patch.dict(os.environ,{'MEMORY_API_KEY':'test','SKIP_MODEL_STARTUP':'true'}):
            with TestClient(api.app, headers={"X-Api-Key": "test"}) as client:
                response=client.post('/config',json={'rank_fusion_enabled':True,'rank_fusion_dense_weight':35,'rank_fusion_rerank_weight':65})
                self.assertEqual(response.status_code,200,response.text)
                self.assertTrue(client.get('/config').json()['rank_fusion_enabled'])
                for value in (True,1.5,-2):
                    self.assertEqual(client.post('/config',json={'rank_fusion_mp_limit':value}).status_code,422)
                response=client.post('/search',json={'user_id':'u','query':'q','top_k':2})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(set(response.json()['data'][0]),{'id','content','score','created_at'})
