
import asyncio
import logging
import os
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from authservice.db import evaluation as evaluation_collection
from authservice.security import get_current_user


# ============================================================
# CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)

router = APIRouter()

JWT_SECRET_KEY = os.getenv("SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

if not JWT_SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing from your environment.")

active_connections: set[WebSocket] = set()


# ============================================================
# USER LEADERBOARD
# ============================================================

async def get_users_board() -> list[dict]:
    pipeline = [
        {
            "$match": {
                "user_id": {"$exists": True},
                "final_score": {"$type": "number"},
            }
        },
        {
            "$group": {
                "_id": "$user_id",
                "username": {"$first": "$username"},
                "best_score": {"$max": "$final_score"},
                "average_score": {"$avg": "$final_score"},
                "total_evaluations": {"$sum": 1},
            }
        },
        {
            "$sort": {
                "best_score": -1,
                "average_score": -1,
                "total_evaluations": -1,
            }
        },
        {"$limit": 100},
    ]

    results = await evaluation_collection.aggregate(pipeline).to_list(length=100)

    return [
        {
            "rank": rank,
            "username": item.get("username") or "Unknown",
            "best_score": round(float(item.get("best_score", 0)), 2),
            "average_score": round(float(item.get("average_score", 0)), 2),
            "total_evaluations": item.get("total_evaluations", 0),
        }
        for rank, item in enumerate(results, start=1)
    ]


# ============================================================
# PROMPT LEADERBOARD
# ============================================================

async def get_prompts_board() -> list[dict]:
    cursor = (
        evaluation_collection.find(
            {"final_score": {"$type": "number"}},
            {
                "_id": 0,
                "username": 1,
                "prompt": 1,
                "final_score": 1,
                "grade": 1,
                "created_at": 1,
            },
        )
        .sort([("final_score", -1), ("created_at", -1)])
        .limit(100)
    )

    results = await cursor.to_list(length=100)
    prompts = []

    for rank, item in enumerate(results, start=1):
        created_at = item.get("created_at")

        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        elif created_at is not None:
            created_at = str(created_at)

        prompts.append(
            {
                "rank": rank,
                "username": item.get("username") or "Unknown",
                "prompt": item.get("prompt", ""),
                "score": round(float(item.get("final_score", 0)), 2),
                "grade": item.get("grade", "N/A"),
                "created_at": created_at,
            }
        )

    return prompts


# ============================================================
# LEADERBOARD DATA
# ============================================================

async def get_leaderboard_data() -> dict:
    users, prompts = await asyncio.gather(
        get_users_board(),
        get_prompts_board(),
    )

    return {
        "type": "leaderboard_update",
        "users": users,
        "prompts": prompts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# REST ENDPOINTS
# ============================================================

@router.get("/leaderboard/users")
async def get_user_leaderboard(
    current_user: dict = Depends(get_current_user),
):
    return {"users": await get_users_board()}


@router.get("/leaderboard/prompts")
async def get_prompt_leaderboard(
    current_user: dict = Depends(get_current_user),
):
    return {"prompts": await get_prompts_board()}


# ============================================================
# LIVE LEADERBOARD BROADCAST
# ============================================================

async def broadcast_leaderboard_update() -> None:
    try:
        data = await get_leaderboard_data()
    except Exception:
        logger.exception("Unable to build leaderboard")
        return

    for websocket in list(active_connections):
        try:
            await websocket.send_json(data)
        except Exception:
            active_connections.discard(websocket)


# ============================================================
# WEBSOCKET LEADERBOARD
# ============================================================

@router.websocket("/leaderboard/live")
async def live_leaderboard(websocket: WebSocket, token: str):
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

        user_id = payload.get("sub") or payload.get("user_id")

        if not user_id:
            await websocket.close(code=1008)
            return

    except jwt.InvalidTokenError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    active_connections.add(websocket)

    try:
        # Send current leaderboard immediately
        await websocket.send_json(await get_leaderboard_data())

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info("Leaderboard disconnected: %s", user_id)

    except Exception:
        logger.exception("Leaderboard WebSocket error")

    finally:
        active_connections.discard(websocket)
