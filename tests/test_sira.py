import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from paimem.sira import SiraConfig, SiraRequest, LLMSiraExpander, filter_phrases, parse_phrases
from paimem.memory_store import SQLiteMemoryStore, MemoryMessage
from paimem.conflict_strategy import JudgmentQueries

class Embedder:
    identity='sira-test'
    def encode_documents(self,texts): return np.asarray([[1.,0.] for _ in texts],dtype=np.float32)
    def encode_queries(self,texts): return self.encode_documents(texts)

class SiraTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.expander=Mock();self.expander.expand.return_value=['target']
    def store(self, **kw):
        kw.setdefault('retrieval_mode','hybrid')
        store=SQLiteMemoryStore(Path(self.tmp.name)/'store.db',Embedder(),sira_expander=self.expander,**kw)
        store.add('r','u','s',[MemoryMessage('user','original'),MemoryMessage('user','target')])
        return store
    def test_default_off_and_runtime_roundtrip(self):
        store=self.store();before=store.search('u','original',2)
        self.expander.expand.assert_not_called()
        store.update_config(sira_enabled=True)
        _,detail=store.search('u','original',2,diagnostics=True)
        self.assertEqual(detail['sira']['kept_count'],1)
        self.assertTrue(detail['sira']['active'])
        store.update_config(sira_enabled=False)
        self.assertEqual(store.search('u','original',2),before)
        self.assertEqual(self.expander.expand.call_count,1)
    def test_timeout_and_filtered_fall_back(self):
        store=self.store();before=store.search('u','original',2)
        store.update_config(sira_enabled=True)
        self.expander.expand.side_effect=TimeoutError('secret')
        hits,detail=store.search('u','original',2,diagnostics=True)
        self.assertEqual(hits,before);self.assertEqual(detail['sira']['status'],'error')
        self.assertNotIn('secret',str(detail))
        self.expander.expand.side_effect=None;self.expander.expand.return_value=['unseen']
        hits,detail=store.search('u','original',2,diagnostics=True)
        self.assertEqual(hits,before);self.assertEqual(detail['sira']['status'],'all_filtered')
    def test_user_isolation_and_no_add_or_empty_calls(self):
        store=self.store(sira_config=SiraConfig(enabled=True))
        self.expander.expand.assert_not_called()
        store.search('absent','original',2);self.expander.expand.assert_not_called()
        store.add('r2','other','s',[MemoryMessage('user','unseen')])
        _,detail=store.search('other','original',2,diagnostics=True)
        self.assertFalse(detail['sira']['active'])
        _,detail=store.search('u','original',2,diagnostics=True)
        self.assertTrue(detail['sira']['active'])
    def test_dense_or_zero_weight_does_not_call(self):
        store=self.store(retrieval_mode='dense',sira_config=SiraConfig(enabled=True))
        store.search('u','original',2);self.expander.expand.assert_not_called()
        store.update_config(retrieval_mode='hybrid',lexical_weight=0)
        store.search('u','original',2);self.expander.expand.assert_not_called()
    def test_conflict_routes_call_once_with_original_query(self):
        strategy=Mock();strategy.classify.return_value=JudgmentQueries('positive','negative','scope')
        strategy.assess.return_value=None
        store=self.store(sira_config=SiraConfig(enabled=True),conflict_strategy=strategy)
        store.search('u','original',10,diagnostics=True)
        self.expander.expand.assert_called_once_with('original',store.sira_config)
    def test_snapshot_survives_toggle_during_request(self):
        store=self.store(sira_config=SiraConfig(enabled=True))
        def expansion(*args):
            store.update_config(sira_enabled=False)
            return ['target']
        self.expander.expand.side_effect=expansion
        _,detail=store.search('u','original',2,diagnostics=True)
        self.assertTrue(detail['sira']['enabled']);self.assertTrue(detail['sira']['active'])
        self.assertFalse(store.get_config()['sira_enabled'])
    def test_missing_client_and_invalid_toggle_are_atomic(self):
        store=self.store();store.sira_expander=None
        before=store.get_config()
        for value in (True,'true',1):
            with self.assertRaises(ValueError):store.update_config(sira_enabled=value,result_window=3)
            self.assertEqual(store.get_config(),before)
    def test_config_validation(self):
        for kw in ({'enabled':1},{'timeout':0},{'max_df_ratio':1.1},{'expansion_weight':float('nan')},{'max_n':0},{'max_n':5},{'max_tokens':True}):
            with self.assertRaises(ValueError):SiraConfig(**kw)
    def test_df_ngram_and_parser(self):
        self.assertEqual(filter_phrases(['red car','red boat','blue car','red plane'],['red','car','red car','unseen'],SiraConfig(max_df_ratio=.25)),['red car'])
        self.assertEqual(parse_phrases('```json\n{"keywords":[" a ",1]}\n```'),['a'])
        self.assertEqual(parse_phrases('invalid'),[])
    def test_llm_fixed_model_original_query_and_timeout(self):
        client=Mock();client.chat.completions.create.return_value.choices=[Mock(message=Mock(content='{"keywords":["target"]}'))]
        result=LLMSiraExpander(client,'fixed').expand('original',SiraConfig(timeout=5))
        kw=client.chat.completions.create.call_args.kwargs
        self.assertEqual(result,['target']);self.assertEqual(kw['model'],'fixed');self.assertEqual(kw['timeout'],5)
        self.assertIn('original',kw['messages'][0]['content']);self.assertEqual(kw['extra_body'],{'enable_thinking':False})
    def test_api_config_and_cli(self):
        from paimem import api,cli
        from fastapi.testclient import TestClient
        store=self.store();api.app.dependency_overrides[api.get_store]=lambda:store
        self.addCleanup(api.app.dependency_overrides.clear)
        with patch.dict(os.environ, {'MEMORY_API_KEY': 'test'}), patch.object(api, 'get_store', return_value=store), TestClient(api.app, headers={'X-Api-Key': 'test'}) as client:
            self.assertFalse(client.get('/config').json()['sira_enabled'])
            self.assertTrue(client.post('/config',json={'sira_enabled':True}).json()['sira_enabled'])
            self.assertEqual(client.post('/config',json={'sira_enabled':'true'}).status_code,422)
            self.assertEqual(client.post('/config',json={'sira_expansion_weight':.8}).status_code,422)
        self.assertIs(cli.RUNTIME_SETTINGS['sira_enabled'],bool)
    def test_env_and_missing_model(self):
        from paimem import api
        with patch.dict(os.environ,{},clear=True),patch.dict(api.STORE_INIT_OVERRIDES,{},clear=True):
            self.assertFalse(api._build_sira_config().enabled)
            self.assertIsNone(api._build_sira_expander(SiraConfig()))
            with self.assertRaises(ValueError):api._build_sira_expander(SiraConfig(enabled=True))
            with patch.dict(os.environ,{'MEMORY_SIRA_ENABLED':'true','MEMORY_SIRA_TIMEOUT':'5'}):
                cfg=api._build_sira_config();self.assertTrue(cfg.enabled);self.assertEqual(cfg.timeout,5)
            with patch.dict(os.environ,{'MEMORY_SIRA_ENABLED':'maybe'}):
                with self.assertRaises(ValueError):api._build_sira_config()

if __name__=='__main__':unittest.main()
