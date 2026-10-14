"""Single-chunk malformed extraction must not mutate a document (#5582).

Uses real PostgreSQL and retain orchestration. Only LLM transport, embeddings,
and reranking are synthetic. This does not assert multi-chunk atomicity.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
from hindsight_api.engine.task_backend import SyncTaskBackend
from tests.test_llm_reasoning_effort_env import DummyCrossEncoder
from tests.test_retain_same_document_concurrency import _StubEmbeddings


@pytest_asyncio.fixture
async def envelope_memory(pg0_db_url, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=DummyCrossEncoder(),
        task_backend=SyncTaskBackend(),
        run_migrations=False,
        skip_llm_verification=True,
    )
    await memory.initialize()
    cfg = memory._config_resolver._global_config
    cfg.enable_auto_consolidation = False
    cfg.retain_batch_enabled = False
    cfg.retain_llm_max_retries = 1
    try:
        yield memory
    finally:
        await memory.close()


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("response", [{}, {"final": {"facts": [{"what": "Bob moved to Berlin."}]}}, {"facts": None}])
async def test_failed_single_chunk_retain_preserves_document(
    envelope_memory, request_context, monkeypatch, existing, response
):
    memory = envelope_memory
    bank = f"test-envelope-{uuid.uuid4().hex}"
    document = "synthetic-document"
    pool = await memory._get_pool()

    async def snapshot():
        result = {}
        for table in ("documents", "chunks", "memory_units"):
            rows = await pool.fetch(
                f"SELECT to_jsonb(t)::text AS row FROM {table} t WHERE bank_id = $1 ORDER BY to_jsonb(t)::text",
                bank,
            )
            result[table] = [row["row"] for row in rows]
        return result

    original_call = MockLLM.call
    calls = 0

    async def call(self, *args, **kwargs):
        nonlocal calls
        if kwargs.get("scope") == "retain_extract_facts":
            calls += 1
            return LLMCallResult(content=response, usage=TokenUsage())
        return await original_call(self, *args, **kwargs)

    try:
        if existing:
            ids = await memory.retain_async(
                bank_id=bank,
                content="Alice works at Acme as a software engineer.",
                document_id=document,
                request_context=request_context,
            )
            assert ids
        before = await snapshot()
        assert bool(before["memory_units"]) == existing
        monkeypatch.setattr(MockLLM, "call", call)
        with pytest.raises(RuntimeError, match="Fact extraction failed"):
            await memory.retain_batch_async(
                bank_id=bank,
                contents=[{"content": "Bob moved to Berlin.", "document_id": document, "force_reextract": True}],
                request_context=request_context,
            )
        assert calls == 2
        assert await snapshot() == before
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_corrected_retry_persists_fact(envelope_memory, request_context, monkeypatch):
    memory = envelope_memory
    bank = f"test-envelope-{uuid.uuid4().hex}"
    original_call = MockLLM.call
    calls = 0

    async def call(self, *args, **kwargs):
        nonlocal calls
        if kwargs.get("scope") == "retain_extract_facts":
            calls += 1
            response = {} if calls == 1 else {"facts": [{"what": "Bob moved to Berlin."}]}
            return LLMCallResult(content=response, usage=TokenUsage())
        return await original_call(self, *args, **kwargs)

    monkeypatch.setattr(MockLLM, "call", call)
    try:
        ids = await memory.retain_async(
            bank_id=bank,
            content="Bob moved to Berlin.",
            document_id="synthetic-document",
            request_context=request_context,
        )
        assert calls == 2
        assert len(ids) == 1
        pool = await memory._get_pool()
        rows = await pool.fetch("SELECT text FROM memory_units WHERE bank_id = $1", bank)
        assert [row["text"] for row in rows] == ["Bob moved to Berlin."]
    finally:
        await memory.delete_bank(bank, request_context=request_context)
