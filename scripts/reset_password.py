"""
Script to reset user password for the Indoor Navigation System
Run from backend directory: python scripts/reset_password.py
Or from project root: python backend/scripts/reset_password.py
"""
import asyncio
import sys
import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from passlib.context import CryptContext

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models import User

# MongoDB connection
MONGODB_URL = "mongodb+srv://manojmahato08779_db_user:ucPCrRk3FwAwwocz@cluster0.2s8wyva.mongodb.net/?appName=Cluster0"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def reset_password(username: str, new_password: str):
    """Reset user password"""
    
    # Initialize MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.indoor_navigation
    
    # Initialize beanie
    await init_beanie(database=database, document_models=[User])
    
    # Find user
    user = await User.find_one(User.username == username)
    
    if not user:
        print(f"❌ User '{username}' not found!")
        return
    
    # Update password
    user.hashed_password = pwd_context.hash(new_password)
    await user.save()
    
    admin_badge = "👑 ADMIN" if user.is_admin else "👤 USER"
    print(f"✅ Password reset successfully!")
    print(f"   User: {username} {admin_badge}")
    print(f"   Email: {user.email}")
    print(f"\nNew login credentials:")
    print(f"   Username: {username}")
    print(f"   Password: {new_password}")

def main():
    print("=" * 60)
    print("Indoor Navigation System - Password Reset")
    print("=" * 60)
    print()
    
    username = input("Enter username: ")
    new_password = input("Enter new password: ")
    
    if not username or not new_password:
        print("❌ Username and password are required!")
        return
    
    confirm = input(f"\nReset password for '{username}'? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("❌ Password reset cancelled.")
        return
    
    asyncio.run(reset_password(username, new_password))

if __name__ == "__main__":
    main()
