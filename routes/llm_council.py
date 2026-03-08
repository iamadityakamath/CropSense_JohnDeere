from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["llm-council"])

# Simple in-memory store for received council payloads.
LLM_COUNCIL_MESSAGES: list[dict[str, Any]] = []


class LLMCouncilRequest(BaseModel):
    data: dict[str, Any]


def forward_to_llm_council(data: dict[str, Any]) -> dict[str, Any]:
    LLM_COUNCIL_MESSAGES.append(data)
    return {
        "status": "accepted",
        "message_count": len(LLM_COUNCIL_MESSAGES),
    }


@router.get("/llm-council")
def llm_council() -> dict[str, Any]:
    return {
        "status": "llm_council route is defined",
        "message_count": len(LLM_COUNCIL_MESSAGES),
    }


@router.post("/llm-council")
def llm_council_ingest(payload: LLMCouncilRequest) -> dict[str, Any]:
    return forward_to_llm_council(payload.data)


@router.get("/llm-council/messages")
def llm_council_messages() -> dict[str, Any]:
    return {
        "count": len(LLM_COUNCIL_MESSAGES),
        "messages": LLM_COUNCIL_MESSAGES,
    }


@router.get("/llm-council/latest")
def llm_council_latest() -> dict[str, Any]:
    if not LLM_COUNCIL_MESSAGES:
        return {"message": "No messages received yet"}

    return {
        "message": "latest llm-council payload",
        "data": LLM_COUNCIL_MESSAGES[-1],
    }
