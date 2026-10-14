"""Malformed envelopes must not look like explicitly empty extractions (#5582)."""

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    RetainContent,
    _coerce_fact_response,
    _extract_facts_from_chunk,
    extract_facts_from_contents_batch_api,
)

BAD_ENVELOPES = [
    {},
    {"final": {"facts": [{"what": "Alice works at Acme."}]}},
    {"facts": None},
    {"facts": {}},
    {"facts": ""},
    {"facts": 0},
    {"facts": "invalid"},
]


def config():
    return dataclasses.replace(
        _get_raw_config(),
        retain_llm_max_retries=1,
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
    )


async def extract(llm):
    return await _extract_facts_from_chunk(
        chunk="Alice works at Acme.",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="",
        llm_config=llm,
        config=config(),
    )


@pytest.mark.parametrize("response", BAD_ENVELOPES)
def test_invalid_envelope_is_not_coerced(response):
    assert _coerce_fact_response(response) is None


@pytest.mark.parametrize("response", [{"facts": []}, [], {"facts": [{"text": "Alice works at Acme."}]}])
def test_supported_envelopes_are_preserved(response):
    assert _coerce_fact_response(response) == (response if isinstance(response, dict) else {"facts": response})


@pytest.mark.asyncio
@pytest.mark.parametrize("response", BAD_ENVELOPES)
async def test_invalid_envelope_retries_then_fails(response):
    llm = MagicMock()
    llm.call = AsyncMock(return_value=LLMCallResult(content=response, usage=TokenUsage()))
    with pytest.raises(RuntimeError, match="Fact extraction failed"):
        await extract(llm)
    assert llm.call.call_count == 2
    hint = llm.call.call_args.kwargs["messages"][-1]
    assert hint["role"] == "user"
    assert '"facts"' in hint["content"]
    assert "list" in hint["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("response", BAD_ENVELOPES)
async def test_corrected_envelope_succeeds_on_retry(response):
    llm = MagicMock()
    llm.call = AsyncMock(
        side_effect=[
            LLMCallResult(content=response, usage=TokenUsage(input_tokens=10)),
            LLMCallResult(content={"facts": [{"what": "Alice works at Acme."}]}, usage=TokenUsage(input_tokens=20)),
        ]
    )
    facts, usage = await extract(llm)
    assert len(facts) == 1
    assert "Alice works at Acme." in facts[0].fact
    assert llm.call.call_count == 2
    assert usage.input_tokens == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("response", BAD_ENVELOPES)
async def test_batch_invalid_envelope_fails_before_returning_empty_result(response):
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
        {
            "custom_id": "chunk_0",
            "response": {
                "body": {
                    "choices": [{"message": {"content": json.dumps(response)}}],
                }
            },
        }
    ]
    llm = MagicMock()
    llm.batch_provider_impl = AsyncMock(return_value=impl)
    with patch(
        "hindsight_api.engine.retain.fact_extraction._write_batch_extraction_errors", new_callable=AsyncMock
    ) as write:
        with pytest.raises(RuntimeError, match="invalid extraction envelope"):
            await extract_facts_from_contents_batch_api(
                contents=[RetainContent(content="Alice works at Acme.")],
                llm_config=llm,
                config=config(),
            )
    errors = write.call_args.args[3]
    assert errors.count == 1
    assert "facts" in errors.sample[0]
    assert "list" in errors.sample[0]
