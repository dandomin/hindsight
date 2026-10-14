"""Real PostgreSQL batch-envelope safety and supported payloads for #5582."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.retain.fact_extraction import RetainContent, extract_facts_from_contents_batch_api
from tests.test_extraction_envelope import BAD_ENVELOPES, config
from tests.test_extraction_envelope_db import envelope_memory  # noqa: F401


def batch_impl(response):
    impl = AsyncMock()
    impl.model = "synthetic-model"
    impl.provider = "openai"
    impl.openai_service_tier = None
    impl.batch_account_key = "synthetic-account"
    impl.submit_batch.return_value = {"batch_id": "synthetic-batch"}
    impl.get_batch_status.return_value = {
        "status": "completed",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
    }
    impl.retrieve_batch_results.return_value = [
        {"custom_id": "chunk_0", "response": {"body": {"choices": [{"message": {"content": json.dumps(response)}}]}}}
    ]
    return impl


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected",
    [
        ({"facts": []}, 0),
        ([], 0),
        ([{"what": "Alice works at Acme."}], 1),
        ({"facts": [{"text": "Alice works at Acme."}]}, 1),
        ({"facts": [{"factual_core": "Alice works at Acme."}]}, 1),
        ({"facts": [{"what": "Alice works at Acme."}]}, 1),
    ],
)
@pytest.mark.parametrize("mode", ["concise", "verbatim"])
async def test_batch_supported_payloads(response, expected, mode):
    llm = MagicMock()
    llm.batch_provider_impl = AsyncMock(return_value=batch_impl(response))
    cfg = config()
    cfg.retain_extraction_mode = mode
    result = await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice works at Acme.")], llm_config=llm, config=cfg
    )
    assert len(result.facts) == expected
    if expected:
        assert result.facts[0].fact_text == "Alice works at Acme."


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("response", BAD_ENVELOPES)
async def test_batch_malformed_preserves_rows(envelope_memory, request_context, monkeypatch, existing, response):
    memory = envelope_memory
    bank = f"test-batch-envelope-{uuid.uuid4().hex}"
    document = "synthetic-document"
    pool = await memory._get_pool()

    async def snapshot():
        result = {}
        for table in ("documents", "chunks", "memory_units"):
            rows = await pool.fetch(
                f"SELECT to_jsonb(t)::text AS row FROM {table} t WHERE bank_id=$1 ORDER BY to_jsonb(t)::text", bank
            )
            result[table] = [row["row"] for row in rows]
        return result

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
        memory._config_resolver._global_config.retain_batch_enabled = True
        monkeypatch.setattr(
            memory._retain_llm_config, "batch_provider_impl", AsyncMock(return_value=batch_impl(response))
        )
        with pytest.raises(RuntimeError, match="invalid extraction envelope"):
            await memory.retain_batch_async(
                bank_id=bank,
                contents=[{"content": "Bob moved to Berlin.", "document_id": document, "force_reextract": True}],
                request_context=request_context,
            )
        assert await snapshot() == before
        # A fresh, corrected batch can recover without retaining any failed replacement.
        monkeypatch.setattr(
            memory._retain_llm_config,
            "batch_provider_impl",
            AsyncMock(return_value=batch_impl({"facts": [{"what": "Bob moved to Berlin."}]})),
        )
        ids = await memory.retain_batch_async(
            bank_id=bank,
            contents=[{"content": "Bob moved to Berlin.", "document_id": document, "force_reextract": True}],
            request_context=request_context,
        )
        assert len(ids[0]) == 1
        rows = await pool.fetch("SELECT text FROM memory_units WHERE bank_id=$1", bank)
        assert [row["text"] for row in rows] == ["Bob moved to Berlin."]
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("response", [{"facts": []}, []])
async def test_explicit_empty_batch_remains_successful(envelope_memory, request_context, monkeypatch, response):
    memory = envelope_memory
    bank = f"test-empty-batch-{uuid.uuid4().hex}"
    memory._config_resolver._global_config.retain_batch_enabled = True
    monkeypatch.setattr(
        memory._retain_llm_config, "batch_provider_impl", AsyncMock(return_value=batch_impl(response))
    )
    try:
        result = await memory.retain_batch_async(
            bank_id=bank,
            contents=[{"content": "No facts here.", "document_id": "synthetic-document"}],
            request_context=request_context,
        )
        assert result == [[]]
        pool = await memory._get_pool()
        assert await pool.fetchval("SELECT count(*) FROM documents WHERE bank_id=$1", bank) == 1
        assert await pool.fetchval("SELECT count(*) FROM memory_units WHERE bank_id=$1", bank) == 0
    finally:
        await memory.delete_bank(bank, request_context=request_context)
