import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, List, Optional, Union

import jwt
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pandas_to_pydantic import dataframe_to_pydantic
from passlib.context import CryptContext
from pydantic import BaseModel

from ml_api.cluster_model import clustering_data

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 1

app = FastAPI()


class Item(BaseModel):
    id: int
    age: int
    sex: str
    job: int
    housing: str
    save_account: str
    check_account: str
    credit_amount: int
    duration: int
    purpose: str
    cluster: int | None = None


class User(BaseModel):
    username: str
    role: str
    email: str
    age: int


class Token(BaseModel):
    access_token: str
    token_type: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

fake_users_db = {
    "john_doe": {
        "username": "john_doe",
        "hashed_password": pwd_context.hash("secret"),
        "email": "john@example.com",
        "age": 25,
        "role": "admin",
    },
}


class TokenData(BaseModel):
    username: Optional[str] = None


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Union[bool, dict]:
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = (
        datetime.now(timezone.utc) + expires_delta
        if expires_delta
        else datetime.now(timezone.utc) + timedelta(minutes=15)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/token/", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires  # type: ignore
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # type: ignore
        print(payload)
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = fake_users_db.get(str(token_data.username))
    if user is None:
        raise credentials_exception
    return user


class RoleChecker:
    def __init__(self, allowed_roles: Any) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, user: Annotated[User, Depends(get_current_user)]) -> bool:
        if user["role"] in self.allowed_roles:  # type: ignore
            return True
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You don't have enough permissions",
        )


@app.post("/cluster/")
async def cluster_data(
    _: Annotated[bool, Depends(RoleChecker(allowed_roles=["admin"]))],
    items: List[Item],
    current_user: User = Depends(get_current_user),
) -> None:
    df = pd.DataFrame([item.model_dump() for item in items])
    _cluster_df = clustering_data(df)
    return dataframe_to_pydantic(_cluster_df, Item)
