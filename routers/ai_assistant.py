"""
AI Assistant API endpoints for auto-generating labels and descriptions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.groq_service import groq_service

router = APIRouter(prefix="/ai", tags=["AI Assistant"])


class LandmarkDescriptionRequest(BaseModel):
    node_type: str
    label: str
    context: Optional[str] = None


class LandmarkDescriptionResponse(BaseModel):
    description: str
    success: bool


class LabelSuggestionRequest(BaseModel):
    node_type: str
    number: Optional[str] = None
    name: Optional[str] = None


class LabelSuggestionResponse(BaseModel):
    label: str
    success: bool


@router.post("/generate-landmark-description", response_model=LandmarkDescriptionResponse)
async def generate_landmark_description(request: LandmarkDescriptionRequest):
    """
    Generate AI-powered landmark description for indoor navigation
    
    - **node_type**: Type of location (room, entrance, elevator, etc.)
    - **label**: Name/label of the location
    - **context**: Optional additional context for better description
    """
    try:
        description = groq_service.generate_landmark_description(
            node_type=request.node_type,
            label=request.label,
            context=request.context
        )
        
        return LandmarkDescriptionResponse(
            description=description,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


@router.post("/suggest-label", response_model=LabelSuggestionResponse)
async def suggest_label(request: LabelSuggestionRequest):
    """
    Get smart label suggestion based on node type
    
    - **node_type**: Type of location
    - **number**: Optional room/floor number
    - **name**: Optional custom name
    """
    try:
        label = groq_service.suggest_label(
            node_type=request.node_type,
            number=request.number,
            name=request.name
        )
        
        return LabelSuggestionResponse(
            label=label,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to suggest label: {str(e)}")


@router.get("/node-types")
async def get_node_types():
    """Get list of available node types with descriptions"""
    return {
        "node_types": [
            {"value": "room", "label": "Room", "icon": "meeting_room", "requires_number": True},
            {"value": "entrance", "label": "Entrance", "icon": "door_front_door", "requires_number": False},
            {"value": "exit", "label": "Exit", "icon": "exit_to_app", "requires_number": False},
            {"value": "elevator", "label": "Elevator", "icon": "elevator", "requires_number": False},
            {"value": "stairs", "label": "Stairs", "icon": "stairs", "requires_number": False},
            {"value": "bathroom", "label": "Restroom", "icon": "wc", "requires_number": False},
            {"value": "cafe", "label": "Café", "icon": "restaurant", "requires_number": False},
            {"value": "office", "label": "Office", "icon": "business", "requires_number": True},
            {"value": "waypoint", "label": "Waypoint", "icon": "location_on", "requires_number": False},
        ]
    }
