from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer
from fastapi.responses import FileResponse
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
from passlib.context import CryptContext
import uuid
import aiofiles
from pathlib import Path
import httpx
from typing import Optional

from models import User
from schemas import UserUpdate, User as UserSchema
from auth_utils import get_current_user
import os

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 16400

# OTPless Configuration
OTPLESS_APP_ID = "1ZL5IUR4FITTIV93TX49"
OTPLESS_CLIENT_ID = "PPLZ8PLRP17J4FALF78B8G6MG90A95LS"
OTPLESS_CLIENT_SECRET = "x7ass763h83f36uo5t2uyx22kquoi2r4"
OTPLESS_API_BASE = "https://api.otpless.app/auth"
OTPLESS_VERIFY_TOKEN_URL = "https://api.otpless.app/auth/userInfo"

# Profile pictures directory
UPLOAD_DIR = Path("uploads/profile_pictures")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

http_bearer = HTTPBearer()

# OTPless request models
class SendOTPRequest(BaseModel):
    channel: str  # "PHONE", "EMAIL", "WHATSAPP"
    phone: Optional[str] = None  # Phone number with country code (e.g., "+919876543210")
    email: Optional[str] = None  # Email address
    countryCode: Optional[str] = "91"  # Default to India

class VerifyOTPRequest(BaseModel):
    channel: str  # "PHONE", "EMAIL", "WHATSAPP"
    phone: Optional[str] = None
    email: Optional[str] = None
    otp: str  # OTP code received
    requestId: Optional[str] = None  # Request ID from send OTP

# Alias for backward compatibility
OTPlessInitiateRequest = SendOTPRequest
OTPlessVerifyRequest = VerifyOTPRequest

class OTPlessLoginRequest(BaseModel):
    channel: str  # "PHONE", "EMAIL", "WHATSAPP"
    phone: Optional[str] = None
    email: Optional[str] = None
    otp: str  # OTP code received

class OTPlessSignupRequest(BaseModel):
    channel: str  # "PHONE", "EMAIL", "WHATSAPP"
    phone: Optional[str] = None
    email: Optional[str] = None
    otp: str  # OTP code received
    username: str  # Required for new user registration
    role: Optional[str] = "user"  # "user" or "admin"

class SocialLoginRequest(BaseModel):
    token: str  # ID token from OTPless (after Google/Facebook login)
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    username: Optional[str] = None  # Optional, will be generated if not provided
    role: Optional[str] = "user"  # "user" or "admin"

class OTPlessResponse(BaseModel):
    success: bool
    message: str
    request_id: Optional[str] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserSchema

class CheckUserRequest(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None

class CheckUserResponse(BaseModel):
    exists: bool
    user: Optional[UserSchema] = None

# Traditional auth models
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ============================================
# OTPless Helper Functions
# ============================================

async def send_otp_via_otpless(channel: str, phone: Optional[str] = None, email: Optional[str] = None, country_code: str = "91") -> dict:
    """
    Send OTP via OTPless API
    Returns: {"success": bool, "request_id": str, "message": str}
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "clientId": OTPLESS_CLIENT_ID,
                "clientSecret": OTPLESS_CLIENT_SECRET,
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": channel,
                "otpLength": 6,
                "expiry": 300  # 5 minutes
            }
            
            if channel in ["PHONE", "WHATSAPP"] and phone:
                # Format phone number
                formatted_phone = phone
                if not phone.startswith('+'):
                    # Remove any non-digit characters
                    clean_phone = ''.join(filter(str.isdigit, phone))
                    # Add country code if not present
                    if not clean_phone.startswith(country_code):
                        clean_phone = country_code + clean_phone
                    formatted_phone = '+' + clean_phone
                
                payload["phoneNumber"] = formatted_phone
                
            elif channel == "EMAIL" and email:
                payload["email"] = email
            else:
                return {"success": False, "message": "Invalid channel or missing contact information"}
            
            response = await client.post(
                f"{OTPLESS_API_BASE}/otp/v1/send",
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            response_data = response.json()
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "request_id": response_data.get("orderId", response_data.get("requestId", "")),
                    "message": "OTP sent successfully"
                }
            else:
                return {
                    "success": False,
                    "message": response_data.get("message", f"Failed to send OTP: {response.text}")
                }
                
    except Exception as e:
        return {
            "success": False,
            "message": f"Error sending OTP: {str(e)}"
        }

async def verify_otp_via_otpless(channel: str, otp: str, phone: Optional[str] = None, email: Optional[str] = None, request_id: Optional[str] = None, country_code: str = "91") -> dict:
    """
    Verify OTP via OTPless API
    Returns: {"success": bool, "message": str, "user_data": dict}
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "clientId": OTPLESS_CLIENT_ID,
                "clientSecret": OTPLESS_CLIENT_SECRET,
                "Content-Type": "application/json"
            }
            
            payload = {
                "otp": otp,
                "channel": channel
            }
            
            if channel in ["PHONE", "WHATSAPP"] and phone:
                # Format phone number
                formatted_phone = phone
                if not phone.startswith('+'):
                    clean_phone = ''.join(filter(str.isdigit, phone))
                    if not clean_phone.startswith(country_code):
                        clean_phone = country_code + clean_phone
                    formatted_phone = '+' + clean_phone
                
                payload["phoneNumber"] = formatted_phone
                
            elif channel == "EMAIL" and email:
                payload["email"] = email
            else:
                return {"success": False, "message": "Invalid channel or missing contact information"}
            
            if request_id:
                payload["orderId"] = request_id
            
            response = await client.post(
                f"{OTPLESS_API_BASE}/otp/v1/verify",
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            response_data = response.json()
            
            if response.status_code == 200:
                if response_data.get("isOTPVerified", False):
                    return {
                        "success": True,
                        "message": "OTP verified successfully",
                        "user_data": {
                            "phone": phone,
                            "email": email,
                            "verified": True
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": response_data.get("message", "Invalid OTP")
                    }
            else:
                return {
                    "success": False,
                    "message": response_data.get("message", f"OTP verification failed: {response.text}")
                }
                
    except Exception as e:
        return {
            "success": False,
            "message": f"Error verifying OTP: {str(e)}"
        }

async def verify_social_token(token: str) -> dict:
    """
    Verify social login token (Google/Facebook) via OTPless
    Returns: {"success": bool, "message": str, "user_data": dict}
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "clientId": OTPLESS_CLIENT_ID,
                "clientSecret": OTPLESS_CLIENT_SECRET,
                "Content-Type": "application/json"
            }
            
            payload = {
                "token": token
            }
            
            response = await client.post(
                OTPLESS_VERIFY_TOKEN_URL,
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract user info from OTPless response
                user_data = {
                    "email": data.get("email"),
                    "phone": data.get("mobile_number") or data.get("phone_number"),
                    "name": data.get("name"),
                    "provider": data.get("identities", [{}])[0].get("identityType", "unknown"),
                    "provider_id": data.get("identities", [{}])[0].get("identityValue"),
                    "verified": True
                }
                
                return {
                    "success": True,
                    "message": "Token verified successfully",
                    "user_data": user_data
                }
            else:
                return {
                    "success": False,
                    "message": f"Token verification failed: {response.text}"
                }
                
    except Exception as e:
        return {
            "success": False,
            "message": f"Error verifying token: {str(e)}"
        }


# ============================================
# Traditional Username/Password Authentication
# ============================================

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Traditional username/password login
    """
    # Find user by username
    user = await User.find_one(User.username == request.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/signup", response_model=LoginResponse)
async def signup(request: SignupRequest):
    """
    Traditional username/password signup
    """
    # Validate username length
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    # Check if username already exists
    existing_user = await User.find_one(User.username == request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if email already exists
    existing_email = await User.find_one(User.email == request.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate password length
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    # Determine if user should be admin
    is_admin = request.role and request.role.lower() == "admin"
    
    # Create new user
    hashed_password = get_password_hash(request.password)
    db_user = User(
        username=request.username,
        email=request.email,
        hashed_password=hashed_password,
        is_admin=is_admin,
        role=request.role.lower() if request.role else "user",
        auth_method="password"
    )
    await db_user.insert()
    
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

# ============================================
# OTPless Authentication Endpoints
# ============================================

@router.post("/send-otp", response_model=OTPlessResponse)
async def send_otp(request: OTPlessInitiateRequest):
    """
    Send OTP via OTPless for phone, email, or WhatsApp authentication
    Supported channels: PHONE, EMAIL, WHATSAPP
    """
    # Validate request
    if request.channel not in ["PHONE", "EMAIL", "WHATSAPP"]:
        raise HTTPException(status_code=400, detail="Channel must be 'PHONE', 'EMAIL', or 'WHATSAPP'")
    
    if request.channel in ["PHONE", "WHATSAPP"] and not request.phone:
        raise HTTPException(status_code=400, detail="Phone number is required for PHONE/WHATSAPP channel")
    
    if request.channel == "EMAIL" and not request.email:
        raise HTTPException(status_code=400, detail="Email is required for EMAIL channel")
    
    # Send OTP
    result = await send_otp_via_otpless(
        channel=request.channel,
        phone=request.phone,
        email=request.email
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return OTPlessResponse(
        success=True,
        message=result["message"],
        request_id=result.get("request_id")
    )

@router.post("/verify-otp", response_model=LoginResponse)
async def verify_otp(request: OTPlessVerifyRequest):
    """
    Verify OTP and login/register user
    - If user exists: Login
    - If user doesn't exist and username provided: Register
    - If user doesn't exist and no username: Error
    """
    # Validate request
    if request.channel not in ["PHONE", "EMAIL", "WHATSAPP"]:
        raise HTTPException(status_code=400, detail="Channel must be 'PHONE', 'EMAIL', or 'WHATSAPP'")
    
    if request.channel in ["PHONE", "WHATSAPP"] and not request.phone:
        raise HTTPException(status_code=400, detail="Phone number is required for PHONE/WHATSAPP channel")
    
    if request.channel == "EMAIL" and not request.email:
        raise HTTPException(status_code=400, detail="Email is required for EMAIL channel")
    
    # Verify OTP
    verification_result = await verify_otp_via_otpless(
        channel=request.channel,
        otp=request.otp,
        phone=request.phone,
        email=request.email
    )
    
    if not verification_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=verification_result["message"]
        )
    
    # Find or create user
    user = None
    contact_identifier = None
    
    if request.channel in ["PHONE", "WHATSAPP"] and request.phone:
        contact_identifier = f"phone:{request.phone}"
        user = await User.find_one(User.email == contact_identifier)
    elif request.channel == "EMAIL" and request.email:
        contact_identifier = request.email
        user = await User.find_one(User.email == contact_identifier)
    
    # If user exists, login
    if user:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }
    
    # If user doesn't exist, register if username provided
    if not request.username:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found. Please provide username to register."
        )
    
    # Validate username
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    # Check if username already exists
    existing_user = await User.find_one(User.username == request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Determine if user should be admin
    is_admin = request.role and request.role.lower() == "admin"
    
    # Create new user
    db_user = User(
        username=request.username,
        email=contact_identifier,
        hashed_password=str(uuid.uuid4()),  # Not used for OTPless auth
        is_admin=is_admin,
        role=request.role.lower() if request.role else "user",
        phone=request.phone if request.channel in ["PHONE", "WHATSAPP"] else None,
        auth_method="otpless"
    )
    await db_user.insert()
    
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@router.post("/social-login", response_model=LoginResponse)
async def social_login(request: SocialLoginRequest):
    """
    Login/Register with Google or Facebook via OTPless
    The token should be obtained from OTPless SDK after social login
    """
    # Verify token with OTPless
    verification_result = await verify_social_token(request.token)
    
    if not verification_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=verification_result["message"]
        )
    
    user_data = verification_result["user_data"]
    
    # Try to find user by email or phone
    user = None
    if user_data.get("email"):
        user = await User.find_one(User.email == user_data["email"])
    elif user_data.get("phone"):
        contact_identifier = f"phone:{user_data['phone']}"
        user = await User.find_one(User.email == contact_identifier)
    
    # If user exists, login
    if user:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }
    
    # If user doesn't exist, register
    if not request.username:
        # Generate username from name or email
        if user_data.get("name"):
            base_username = user_data["name"].lower().replace(" ", "_")
        elif user_data.get("email"):
            base_username = user_data["email"].split("@")[0]
        else:
            base_username = f"user_{uuid.uuid4().hex[:8]}"
        
        # Ensure username is unique
        username = base_username
        counter = 1
        while await User.find_one(User.username == username):
            username = f"{base_username}{counter}"
            counter += 1
        
        request.username = username
    
    # Validate username
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    # Check if username already exists
    existing_user = await User.find_one(User.username == request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Determine if user should be admin
    is_admin = request.role and request.role.lower() == "admin"
    
    # Determine contact identifier
    contact_identifier = user_data.get("email")
    if not contact_identifier and user_data.get("phone"):
        contact_identifier = f"phone:{user_data['phone']}"
    if not contact_identifier:
        contact_identifier = f"{user_data['provider']}:{user_data['provider_id']}"
    
    # Create new user
    db_user = User(
        username=request.username,
        email=contact_identifier,
        hashed_password=str(uuid.uuid4()),  # Not used for social auth
        is_admin=is_admin,
        role=request.role.lower() if request.role else "user",
        phone=user_data.get("phone"),
        auth_method=f"social_{user_data.get('provider', 'unknown')}"
    )
    await db_user.insert()
    
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

# ============================================
# User Profile Management
# ============================================

@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/me", response_model=UserSchema)
async def update_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Update current user's profile (username, email)
    """
    update_data = {}
    
    # Update username if provided
    if user_update.username and user_update.username != current_user.username:
        # Check if username already exists
        existing_user = await User.find_one(User.username == user_update.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
        update_data["username"] = user_update.username
    
    # Update email if provided
    if user_update.email and user_update.email != current_user.email:
        # Check if email already exists
        existing_email = await User.find_one(User.email == user_update.email)
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")
        update_data["email"] = user_update.email
    
    # Apply updates if any
    if update_data:
        for key, value in update_data.items():
            setattr(current_user, key, value)
        await current_user.save()
    
    return current_user


@router.post("/me/profile-picture", response_model=UserSchema)
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a profile picture for the current user
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Generate unique filename
    unique_filename = f"{current_user.id}_{uuid.uuid4().hex}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Delete old profile picture if exists
    if current_user.profile_picture:
        old_file = UPLOAD_DIR / Path(current_user.profile_picture).name
        if old_file.exists():
            old_file.unlink()
    
    # Save new file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Update user profile picture URL
    current_user.profile_picture = f"/auth/profile-picture/{unique_filename}"
    await current_user.save()
    
    return current_user

@router.get("/profile-picture/{filename}")
async def get_profile_picture(filename: str):
    """
    Get a profile picture by filename
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Profile picture not found")
    
    return FileResponse(file_path)

@router.delete("/me/profile-picture", response_model=UserSchema)
async def delete_profile_picture(
    current_user: User = Depends(get_current_user)
):
    """
    Delete the current user's profile picture
    """
    if current_user.profile_picture:
        # Delete file from disk
        file_path = UPLOAD_DIR / Path(current_user.profile_picture).name
        if file_path.exists():
            file_path.unlink()
        
        # Update user
        current_user.profile_picture = None
        await current_user.save()
    
    return current_user
