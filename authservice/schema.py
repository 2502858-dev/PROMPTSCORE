from pydantic import BaseModel, EmailStr, model_validator


class UserRegistration(BaseModel):
    name: str
    email: EmailStr
    confirm_email: EmailStr
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_matches(self):
        if self.email != self.confirm_email:
            raise ValueError("Email and confirm email do not match")
        if self.password != self.confirm_password:
            raise ValueError("Password and confirm password do not match")
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters")
        return self


class login_user(BaseModel):
    email: EmailStr
    password: str
    role: str = "patient"   # "patient" or "doctor" — decides which collection is checked


class UserResponse(BaseModel):
    user_id: str
    name: str
    email: str