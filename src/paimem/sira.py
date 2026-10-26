"""Optional query-side SIRA, adapted from facebookresearch/sira
57f9cbbe5d0a00c91a60f14f3532c8268edd6332 (MIT).
See third_party/licenses/SIRA-MIT.txt and docs/strategies/SIRA.md for adaptation details.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import time


@dataclass(frozen=True)
class SiraConfig:
    enabled: bool = False
    expansion_weight: float = .5
    max_df_ratio: float = .05
    max_n: int = 4
    timeout: float = 30.
    max_tokens: int = 1024

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('sira_enabled must be a boolean')
        for name in ('expansion_weight', 'max_df_ratio', 'timeout'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'sira_{name} must be finite and nonnegative')
        if self.timeout <= 0 or self.max_df_ratio > 1:
            raise ValueError('sira_timeout must be positive; max_df_ratio must be <= 1')
        for name in ('max_n', 'max_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'sira_{name} must be a positive integer')
        if self.max_n > 4:
            raise ValueError('sira_max_n must be <= 4')

    def settings(self):
        return {f'sira_{k}': v for k, v in asdict(self).items()}


# Copyright (c) Meta Platforms, Inc. and affiliates. MIT licensed prompt/parser.
PROMPT = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
You are expanding a search query to improve retrieval. Generate NEW terms that are NOT already present in the query.

Query:
{doc_text}

Your task: identify vocabulary that a correct answer document would contain but that is MISSING from the query — synonyms, domain terms, abbreviations, related concepts, or alternate phrasings the answer document would use.

Rules:
- Do NOT include any word or phrase already present in the query text above.
- Generate at most 10 phrases, 1-{max_n} words each.
- Each phrase must add new vocabulary not already in the query.
- Be DIVERSE: cover multiple distinct facets of what a correct answer would address — do not generate synonyms for the same concept.

Think shortly, not more than 10 sentences. Then output only:
{{"keywords": ["...", "..."]}}
'''


def parse_phrases(raw):
    for start, end in [('{', '}'), ('[', ']')]:
        a, b = raw.find(start), raw.rfind(end)
        if a < 0 or b <= a:
            continue
        try:
            parsed = json.loads(raw[a:b + 1])
            if isinstance(parsed, dict):
                parsed = parsed.get('keywords', [])
            if isinstance(parsed, list):
                return [p.strip() for p in parsed if isinstance(p, str) and p.strip()][:10]
        except json.JSONDecodeError:
            pass
    return []


class LLMSiraExpander:
    def __init__(self, client, model):
        self.client, self.model = client, model

    def expand(self, query, config):
        response = self.client.chat.completions.create(
            model=self.model, messages=[{'role': 'user', 'content': PROMPT.format(doc_text=query, max_n=config.max_n)}],
            max_tokens=config.max_tokens, temperature=.4, seed=42, timeout=config.timeout,
            extra_body={'enable_thinking': False})
        return parse_phrases(response.choices[0].message.content or '')


def filter_phrases(documents, phrases, config):
    from .retrieval import lexical_tokens
    df = Counter()
    for document in documents:
        tokens = lexical_tokens(document)
        df.update(set(tuple(tokens[i:i+n]) for n in range(1, config.max_n+1)
                      for i in range(len(tokens)-n+1)))
    max_df = int(len(documents) * config.max_df_ratio)
    kept = []
    for phrase in phrases:
        tokens = lexical_tokens(phrase)
        if any(0 < df[tuple(tokens[i:i+n])] and
               (max_df == 0 or df[tuple(tokens[i:i+n])] <= max_df)
               for n in range(1, min(config.max_n, len(tokens))+1)
               for i in range(len(tokens)-n+1)):
            kept.append(phrase)
    return kept


class SiraRequest:
    """Request-local, lazy expansion: no cross-user state or persistent caches.

    One original query expansion is reused by conflict routes. Only lexical
    scoring consumes it; embeddings and reranking retain their existing inputs.
    """
    def __init__(self, query, config, expander):
        self.query, self.config, self.expander = query, config, expander
        self.phrases = None
        self.detail = dict(enabled=config.enabled, active=False, status='not_run' if config.enabled else 'disabled',
                           expansion_ms=0., proposed_count=0, kept_count=0)

    def score(self, documents, original):
        from .retrieval import bm25_scores
        if not self.config.enabled:
            return original
        if self.config.expansion_weight == 0:
            self.detail['status'] = 'zero_weight'
            return original
        if self.phrases is None:
            start = time.perf_counter()
            try:
                if self.expander is None:
                    self.detail['status'] = 'unconfigured'
                    self.phrases = []
                else:
                    proposed = self.expander.expand(self.query, self.config)
                    self.phrases = [p.strip() for p in proposed if isinstance(p, str) and p.strip()][:10]
                    self.detail['status'] = 'ok' if self.phrases else 'no_phrases'
            except Exception as exc:
                self.phrases = []
                self.detail.update(status='error', error_type=type(exc).__name__)
            self.detail['expansion_ms'] = (time.perf_counter()-start)*1000
            self.detail['proposed_count'] = len(self.phrases)
        kept = filter_phrases(documents, self.phrases, self.config) if self.phrases else []
        self.detail.update(active=bool(kept), kept_count=len(kept))
        if self.phrases and not kept:
            self.detail['status'] = 'all_filtered'
        elif kept:
            self.detail['status'] = 'ok'
        return original + self.config.expansion_weight * bm25_scores(documents, ' '.join(kept)) if kept else original
