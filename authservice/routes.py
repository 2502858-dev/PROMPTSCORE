from fastapi import APIRouter, HTTPException, status
from authservice.db import collection1

from authservice.schema import UserRegistration, login_user, UserResponse
from authservice.security import create_access_token
from datetime import datetime

from pwdlib import PasswordHash

import bcrypt
import logging

logger = logging.getLogger(__name__)
app = APIRouter()
password_hash = PasswordHash.recommended()


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


@app.post("/register", response_model=UserResponse)
async def register_user(user: UserRegistration):
    """Patient registration only. Doctors register via the separate
    /doctors/apply flow (with admin verification), not this endpoint."""
    try:
        existing_user = await collection1.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        user_dict = user.model_dump()
        user_dict.pop("confirm_email", None)
        user_dict.pop("confirm_password", None)

        user_dict["created_at"] = datetime.utcnow()
        user_dict["is_active"] = True
        user_dict["password"] = get_password_hash(user.password)

        result = await collection1.insert_one(user_dict)
        logger.info(f"New patient registered: email={user.email}")

        return UserResponse(
            user_id=str(result.inserted_id),
            name=user.name,
            email=user.email
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Register ERROR] email={user.email} error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@app.post("/login")
async def login(user: login_user):
    """Role-aware login. role='patient' checks collection1 (uses pwdlib hashing).
    role='doctor' checks doctors_collection (uses bcrypt hashing, from the
    /doctors/apply flow) and additionally requires status == 'verified'."""
    try:
    
      
            db_user = await collection1.find_one({"email": user.email})
            if not db_user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

            if not verify_password(user.password, db_user["password"]):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

            if not db_user.get("is_active", True):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

            token = create_access_token({
                "user_id": str(db_user["_id"]),
                "email": db_user["email"],
                "role": "patient",
            })

            logger.info(f"Patient logged in: email={user.email}")
            return {
                "token": token,
                "role": "patient",
                "patient_id": str(db_user["_id"]),
                "name": db_user["name"],
            }

       

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Login ERROR] email={user.email} error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed. Please try again.")