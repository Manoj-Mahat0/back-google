from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
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

# MongoDB connection
MONGODB_URL = "mongodb+srv://manojmahato08779_db_user:ucPCrRk3FwAwwocz@cluster0.2s8wyva.mongodb.net/?appName=Cluster0"

@app.on_event("startup")
async def startup_event():
    # Initialize MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.indoor_navigation
    
    # Initialize beanie with the models
    await init_beanie(
        database=database,
        document_models=[User, Building, Floor, Room, Waypoint, NavigationPath, ARMarker, BuildingGraph, IndoorGraph]
    )

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(buildings.router, prefix="/buildings", tags=["buildings"])
app.include_router(navigation.router, prefix="/navigation", tags=["navigation"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(offline.router, prefix="/offline", tags=["offline"])
app.include_router(indoor_graph.router, prefix="/indoor", tags=["indoor-navigation"])
app.include_router(ai_assistant.router, prefix="/ai", tags=["ai-assistant"])

@app.get("/")
async def root():
    return {"message": "Indoor Navigation API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)