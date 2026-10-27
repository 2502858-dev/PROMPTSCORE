import logging
import os
import smtplib

from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from authservice.db import evaluation
from authservice.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/challenges",
    tags=["Challenges"],
)

# Store challenge data separately from prompt evaluations.
challenge_records = evaluation.database["challenge_records"]


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChallengeSubmission(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    roll_no: str = Field(min_length=1, max_length=50)
    class_name: str = Field(min_length=1, max_length=80)
    section: str = Field(min_length=1, max_length=30)

    challenge_type: Literal["speed_prompt", "prompt_master"]
    challenge_title: str = Field(min_length=1, max_length=100)

    scores: list[float] = Field(min_length=1, max_length=20)
    elapsed_seconds: int = Field(ge=0, le=120)

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, scores):
        if any(score < 0 or score > 10 for score in scores):
            raise ValueError("Scores must be between 0 and 10.")
        return scores

    @field_validator("name", "roll_no", "class_name", "section")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("This field cannot be empty.")

        return value


# --------------------------------------------------
# EMAIL NOTIFICATION
# --------------------------------------------------

def send_high_score_email(record: dict) -> bool:
    recipient = os.getenv("ADMIN_EMAIL", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not all([
        recipient,
        smtp_host,
        smtp_user,
        smtp_password,
        smtp_from,
    ]):
        logger.warning("SMTP is not configured.")
        return False

    message = EmailMessage()
    message["Subject"] = (
        f"PromptScore AI: New Personal Best - "
        f"{record['name']}"
    )
    message["From"] = smtp_from
    message["To"] = recipient

    message.set_content(
        "A new PromptScore AI challenge result was recorded.\n\n"
        f"Student: {record['name']}\n"
        f"Roll number: {record['roll_no']}\n"
        f"Class: {record['class_name']}\n"
        f"Section: {record['section']}\n"
        f"Challenge: {record['challenge_title']}\n"
        f"Best score: {record['best_score']}/10\n"
        f"Average score: {record['last_average_score']}/10\n"
        f"Attempts: {record['attempts']}\n"
        f"Completed target: {record['last_won']}\n"
        f"Recorded at: {record['updated_at'].isoformat()}\n"
    )

    try:
        with smtplib.SMTP(
            smtp_host,
            smtp_port,
            timeout=15,
        ) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(message)

        return True

    except Exception:
        logger.exception("Failed to send challenge notification.")
        return False


# --------------------------------------------------
# SUBMIT CHALLENGE RESULT
# --------------------------------------------------

@router.post("/submit")
def submit_challenge(
    payload: ChallengeSubmission,
    current_user=Depends(get_current_user),
):
    scores = [float(score) for score in payload.scores]
    average_score = sum(scores) / len(scores)

    if payload.challenge_type == "speed_prompt":
        score_for_record = max(scores)

        won = (
            score_for_record >= 8.0
            and payload.elapsed_seconds <= 60
        )

    else:
        score_for_record = average_score

        won = (
            len(scores) >= 2
            and average_score >= 9.0
            and payload.elapsed_seconds <= 120
        )

    now = datetime.now(timezone.utc)

    # Identify the student's record for this challenge.
    identity = {
        "roll_no_normalized": payload.roll_no.casefold(),
        "class_normalized": payload.class_name.casefold(),
        "section_normalized": payload.section.casefold(),
        "challenge_type": payload.challenge_type,
    }

    previous = challenge_records.find_one(identity)

    previous_best = (
        float(previous.get("best_score", -1))
        if previous
        else -1.0
    )

    is_new_personal_best = score_for_record > previous_best

    if previous:
        update_fields = {
            "name": payload.name,
            "roll_no": payload.roll_no,
            "class_name": payload.class_name,
            "section": payload.section,
            "challenge_title": payload.challenge_title,
            "last_score": round(score_for_record, 2),
            "last_average_score": round(average_score, 2),
            "last_scores": scores,
            "last_won": won,
            "last_elapsed_seconds": payload.elapsed_seconds,
            "updated_at": now,
        }

        if is_new_personal_best:
            update_fields.update({
                "best_score": round(score_for_record, 2),
                "best_scores": scores,
                "best_average_score": round(average_score, 2),
                "best_won": won,
            })

        challenge_records.update_one(
            identity,
            {
                "$set": update_fields,
                "$inc": {"attempts": 1},
            },
        )

        record = challenge_records.find_one(identity)

    else:
        record = {
            **identity,
            "name": payload.name,
            "roll_no": payload.roll_no,
            "class_name": payload.class_name,
            "section": payload.section,
            "challenge_type": payload.challenge_type,
            "challenge_title": payload.challenge_title,
            "best_score": round(score_for_record, 2),
            "best_scores": scores,
            "best_average_score": round(average_score, 2),
            "best_won": won,
            "last_score": round(score_for_record, 2),
            "last_average_score": round(average_score, 2),
            "last_scores": scores,
            "last_won": won,
            "last_elapsed_seconds": payload.elapsed_seconds,
            "attempts": 1,
            "created_at": now,
            "updated_at": now,
        }

        challenge_records.insert_one(record)

    email_sent = False

    if is_new_personal_best:
        email_sent = send_high_score_email(record)

    return {
        "ok": True,
        "message": (
            "New personal best saved."
            if is_new_personal_best
            else "Challenge result saved."
        ),
        "best_score": record["best_score"],
        "is_new_personal_best": is_new_personal_best,
        "email_sent": email_sent,
    }


# --------------------------------------------------
# CHALLENGE LEADERBOARD
# --------------------------------------------------

@router.get("/leaderboard")
def challenge_leaderboard(
    current_user=Depends(get_current_user),
):
    rows = (
        challenge_records.find(
            {},
            {
                "_id": 0,
                "name": 1,
                "roll_no": 1,
                "class_name": 1,
                "section": 1,
                "challenge_type": 1,
                "challenge_title": 1,
                "best_score": 1,
                "best_average_score": 1,
                "attempts": 1,
                "best_won": 1,
            },
        )
        .sort([
            ("best_score", -1),
            ("best_average_score", -1),
        ])
        .limit(100)
    )

    return {"results": list(rows)}