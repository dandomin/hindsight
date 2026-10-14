"""Malformed batch envelopes must fail worker operations with safe DB diagnostics."""

import json
import uuid
from unittest.mock import AsyncMock

import pytest

from hindsight_api.worker import WorkerPoller
from hindsight_api.worker.poller import ClaimedTask
from tests.test_batch_envelope_safety import batch_impl
from tests.test_extraction_envelope_db import envelope_memory  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_malformed_batch_worker_records_failure(envelope_memory, request_context, monkeypatch):
    memory = envelope_memory
    bank = f"test-batch-operation-{uuid.uuid4().hex}"
    operation = uuid.uuid4()
    pool = await memory._get_pool()
    try:
        await pool.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $1)", bank)
        await pool.execute(
            "INSERT INTO async_operations (operation_id, bank_id, operation_type, status) VALUES ($1, $2, 'retain', 'processing')",
            operation,
            bank,
        )
        memory._config_resolver._global_config.retain_batch_enabled = True
        monkeypatch.setattr(
            memory._retain_llm_config,
            "batch_provider_impl",
            AsyncMock(return_value=batch_impl({"final": {"facts": [{"what": "PRIVATE_MODEL_TEXT"}]}})),
        )
        task = ClaimedTask(
            operation_id=str(operation),
            schema=None,
            task_dict={
                "type": "batch_retain",
                "operation_type": "retain",
                "operation_id": str(operation),
                "bank_id": bank,
                "_retry_count": 100,
                "contents": [{"content": "Alice works at Acme.", "document_id": "synthetic-document"}],
            },
        )
        poller = WorkerPoller(
            backend=await memory._get_backend(), worker_id="envelope-test", executor=memory.execute_task
        )
        await poller._execute_task_inner(task)
        row = await pool.fetchrow(
            "SELECT status, error_message, result_metadata FROM async_operations WHERE operation_id=$1", operation
        )
        assert row["status"] == "failed"
        assert "invalid extraction envelope" in row["error_message"]
        metadata = (
            json.loads(row["result_metadata"]) if isinstance(row["result_metadata"], str) else row["result_metadata"]
        )
        assert metadata["extraction_errors_count"] == 1
        assert "expected a top-level 'facts' list" in metadata["extraction_errors_sample"][0]
        assert "PRIVATE_MODEL_TEXT" not in json.dumps(dict(row))
        for table in ("documents", "chunks", "memory_units"):
            assert await pool.fetchval(f"SELECT count(*) FROM {table} WHERE bank_id=$1", bank) == 0
    finally:
        await memory.delete_bank(bank, request_context=request_context)
