"""
Script to create an admin user for the Indoor Navigation System
Run from backend directory: python scripts/create_admin.py
Or from project root: python backend/scripts/create_admin.py
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

async def create_admin_user(username: str, email: str, password: str):
    """Create an admin user"""
    
    # Initialize MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.indoor_navigation
    
    # Initialize beanie
    await init_beanie(database=database, document_models=[User])
    
    # Check if user already exists
    existing_user = await User.find_one(User.username == username)
    if existing_user:
        print(f"❌ User '{username}' already exists!")
        
        # Ask if they want to make existing user admin
        response = input(f"Do you want to make '{username}' an admin? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            existing_user.is_admin = True
            await existing_user.save()
            print(f"✅ User '{username}' is now an admin!")
        return
    
    # Create new admin user
    hashed_password = pwd_context.hash(password)
    admin_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_admin=True
    )
    
    await admin_user.insert()
    print(f"✅ Admin user '{username}' created successfully!")
    print(f"   Email: {email}")
    print(f"   Admin: Yes")
    print(f"\nYou can now login with:")
    print(f"   Username: {username}")
    print(f"   Password: {password}")

async def list_users():
    """List all users"""
    
    # Initialize MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.indoor_navigation
    
    # Initialize beanie
    await init_beanie(database=database, document_models=[User])
    
    users = await User.find_all().to_list()
    
    if not users:
        print("No users found in database.")
        return
    
    print("\n📋 Current Users:")
    print("-" * 60)
    for user in users:
        admin_badge = "👑 ADMIN" if user.is_admin else "👤 USER"
        print(f"{admin_badge} | {user.username:20} | {user.email}")
    print("-" * 60)

def main():
    print("=" * 60)
    print("Indoor Navigation System - Admin User Creator")
    print("=" * 60)
    print()
    
    # Ask what to do
    print("What would you like to do?")
    print("1. Create new admin user")
    print("2. List all users")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\n📝 Create New Admin User")
        print("-" * 60)
        username = input("Enter username: ")
        email = input("Enter email: ")
        password = input("Enter password: ")
        
        if not username or not email or not password:
            print("❌ All fields are required!")
            return
        
        asyncio.run(create_admin_user(username, email, password))
    
    elif choice == "2":
        asyncio.run(list_users())
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
