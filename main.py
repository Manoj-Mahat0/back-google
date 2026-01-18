from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from models import User, Building, Floor, Room, Waypoint, NavigationPath, ARMarker, BuildingGraph, IndoorGraph
from routers import buildings, navigation, auth, admin, offline, indoor_graph, ai_assistant
from auth_utils import get_current_user

app = FastAPI(
    title="Indoor Navigation API",
    description="Backend for AR/VR Indoor Navigation App",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection - use environment variable or fallback to hardcoded
MONGODB_URL = os.getenv(
    "DATABASE_URL",
    "mongodb+srv://manojmahato08779_db_user:ucPCrRk3FwAwwocz@cluster0.2s8wyva.mongodb.net/?appName=Cluster0"
)

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize MongoDB connection
        client = AsyncIOMotorClient(MONGODB_URL)
        database = client.indoor_navigation
        
        # Initialize beanie with the models
        await init_beanie(
            database=database,
            document_models=[User, Building, Floor, Room, Waypoint, NavigationPath, ARMarker, BuildingGraph, IndoorGraph]
        )
        print("✅ Database connected successfully")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        print("⚠️ App will continue but database operations will fail")

# Health check endpoint (must be before other routes for priority)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Indoor Navigation API is running"}

@app.get("/")
async def root():
    return {"message": "Indoor Navigation API", "version": "1.0.0", "status": "online"}

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(buildings.router, prefix="/buildings", tags=["buildings"])
app.include_router(navigation.router, prefix="/navigation", tags=["navigation"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(offline.router, prefix="/offline", tags=["offline"])
app.include_router(indoor_graph.router, prefix="/indoor", tags=["indoor-navigation"])
app.include_router(ai_assistant.router, prefix="/ai", tags=["ai-assistant"])

# This block is not used on Render, but useful for local development
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
