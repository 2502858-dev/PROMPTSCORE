
from typing import Optional
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel

from authservice.security import get_current_user
from ragbuild.graph import Graph_manager
from authservice.db import evaluation

app = APIRouter()


class EvaluateRequest(BaseModel):
    prompt: str
    intended_purpose: Optional[str] = None
    task_description: Optional[str] = None


@app.post("/rag/evaluate")
async def evaluate_prompt(
    request: EvaluateRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        manager = Graph_manager()

        initial_state = {
            "question": request.prompt,
            "intended_purpose": request.intended_purpose or "Not specified",
            "task_description": request.task_description or "Not specified",
        }

        result = await manager.run(initial_state)

        if not isinstance(result, dict):
            raise ValueError("Graph did not return a dictionary")

        evaluation_result = result.get("evaluation", result)

        if not isinstance(evaluation_result, dict):
            raise ValueError("Evaluation result must be a dictionary")

        score = evaluation_result.get(
            "final_score", evaluation_result.get("score", 0)
        )

        dimension_scores = evaluation_result.get(
            "scores", evaluation_result.get("dimension_scores", {})
        )

        evaluation_data = {
            "success": True,
            "final_score": score,
            "score": score,
            "grade": evaluation_result.get("grade", ""),
            "reason": evaluation_result.get(
                "reason", evaluation_result.get("explanation", "")
            ),
            "explanation": evaluation_result.get(
                "explanation", evaluation_result.get("reason", "")
            ),
            "improved_prompt": evaluation_result.get("improved_prompt", ""),
            "scores": dimension_scores,
            "dimension_scores": dimension_scores,
            "strengths": evaluation_result.get("strengths", []),
            "weaknesses": evaluation_result.get("weaknesses", []),
            "intended_purpose": evaluation_result.get(
                "intended_purpose",
                request.intended_purpose or "Not specified",
            ),
        }

        document = {
            "user_id": str(
                current_user.get("_id", current_user.get("id", ""))
            ),
            "username": (
                current_user.get("username")
                or current_user.get("name")
                or current_user.get("email")
                or "Unknown"
            ),
            "prompt": request.prompt,
            "intended_purpose": request.intended_purpose or "Not specified",
            "task_description": request.task_description or "Not specified",
            "final_score": score,
            "grade": evaluation_data["grade"],
            "evaluation": evaluation_data,
            "created_at": datetime.now(timezone.utc),
        }

        saved = await evaluation.insert_one(document)

        return {
            **evaluation_data,
            "evaluation_id": str(saved.inserted_id),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG evaluation or database save failed: {str(e)}",
        )

