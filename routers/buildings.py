from fastapi import APIRouter, Depends, HTTPException
from typing import List
from bson import ObjectId
from datetime import datetime
from beanie.operators import In

from models import Building, Floor, Room, Waypoint, User
from schemas import (
    BuildingCreate, Building as BuildingSchema,
    FloorCreate, Floor as FloorSchema,
    RoomCreate, Room as RoomSchema,
    WaypointCreate, Waypoint as WaypointSchema,
    BuildingStructure, BuildingStructureResponse
)
from auth_utils import get_current_user, get_admin_user
from models import BuildingGraph

from schemas import (
    NavigationGraphRequest,
    GraphNodeResponse
)


router = APIRouter()

@router.get("/", response_model=List[BuildingSchema])
async def get_buildings():
    buildings = await Building.find_all().to_list()
    return buildings

@router.get("/{building_id}", response_model=BuildingSchema)
async def get_building(building_id: str):
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    return building

@router.post("/", response_model=BuildingSchema)
async def create_building(
    building: BuildingCreate, 
    current_user: User = Depends(get_admin_user)
):
    db_building = Building(**building.dict(), created_by=current_user.id)
    await db_building.insert()
    return db_building

@router.get("/{building_id}/floors", response_model=List[FloorSchema])
async def get_building_floors(building_id: str):
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
    return floors

@router.post("/{building_id}/floors", response_model=FloorSchema)
async def create_floor(
    building_id: str,
    floor: FloorCreate,
    current_user: User = Depends(get_admin_user)
):
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    # Verify building exists
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    db_floor = Floor(**floor.dict())
    await db_floor.insert()
    return db_floor

@router.get("/floors/{floor_id}/rooms", response_model=List[RoomSchema])
async def get_floor_rooms(floor_id: str):
    if not ObjectId.is_valid(floor_id):
        raise HTTPException(status_code=400, detail="Invalid floor ID")
    
    rooms = await Room.find(Room.floor_id == ObjectId(floor_id)).to_list()
    return rooms

@router.post("/floors/{floor_id}/rooms", response_model=RoomSchema)
async def create_room(
    floor_id: str,
    room: RoomCreate,
    current_user: User = Depends(get_admin_user)
):
    if not ObjectId.is_valid(floor_id):
        raise HTTPException(status_code=400, detail="Invalid floor ID")
    
    db_room = Room(**room.dict())
    await db_room.insert()
    return db_room

@router.get("/floors/{floor_id}/waypoints", response_model=List[WaypointSchema])
async def get_floor_waypoints(floor_id: str):
    if not ObjectId.is_valid(floor_id):
        raise HTTPException(status_code=400, detail="Invalid floor ID")
    
    waypoints = await Waypoint.find(Waypoint.floor_id == ObjectId(floor_id)).to_list()
    return waypoints

@router.post("/floors/{floor_id}/waypoints", response_model=WaypointSchema)
async def create_waypoint(
    floor_id: str,
    waypoint: WaypointCreate,
    current_user: User = Depends(get_admin_user)
):
    if not ObjectId.is_valid(floor_id):
        raise HTTPException(status_code=400, detail="Invalid floor ID")
    
    db_waypoint = Waypoint(**waypoint.dict())
    await db_waypoint.insert()
    return db_waypoint

@router.post("/{building_id}/structure")
async def create_building_structure(
    building_id: str,
    structure: BuildingStructure,
    current_user: User = Depends(get_admin_user)
):
    """Create complete building structure from GPS coordinates"""
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Create rooms
    for room_data in structure.rooms:
        db_room = Room(**room_data.dict())
        await db_room.insert()
    
    # Create waypoints
    for waypoint_data in structure.waypoints:
        db_waypoint = Waypoint(**waypoint_data.dict())
        await db_waypoint.insert()
    
    return {"message": "Building structure created successfully"}

@router.put("/{building_id}/boundary", response_model=BuildingSchema)
async def update_building_boundary(
    building_id: str,
    boundary_data: dict,
    current_user: User = Depends(get_admin_user)
):
    """Update building boundary points"""
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Update building with new boundary data
    if 'latitude' in boundary_data:
        building.latitude = boundary_data['latitude']
    if 'longitude' in boundary_data:
        building.longitude = boundary_data['longitude']
    if 'boundary_points' in boundary_data:
        building.boundary_points = boundary_data['boundary_points']
    
    await building.save()
    return building

@router.get("/{building_id}/structure", response_model=BuildingStructureResponse)
async def get_building_structure(building_id: str):
    """Get complete building structure including floors, rooms, and waypoints"""
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Get all floors
    floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
    
    floor_ids = [floor.id for floor in floors]
    
    rooms = []
    waypoints = []
    
    if floor_ids:
        rooms = await Room.find(In(Room.floor_id, floor_ids)).to_list()
        waypoints = await Waypoint.find(In(Waypoint.floor_id, floor_ids)).to_list()
    
    # Map rooms to response format
    room_schemas = []
    for r in rooms:
        room_schemas.append(RoomSchema(
            id=str(r.id),
            floor_id=str(r.floor_id),
            name=r.name,
            room_type=r.room_type,
            coordinates=r.coordinates
        ))
        
    # Map waypoints to response format
    waypoint_schemas = []
    for w in waypoints:
        waypoint_schemas.append(WaypointSchema(
            id=str(w.id),
            floor_id=str(w.floor_id),
            latitude=w.latitude,
            longitude=w.longitude,
            floor_number=w.floor_number,
            waypoint_type=w.waypoint_type,
            name=w.name,
            room_name=w.room_name if hasattr(w, 'room_name') else None,
            notes=w.notes if hasattr(w, 'notes') else None,
            images=w.images if hasattr(w, 'images') else None
        ))
        
    return BuildingStructureResponse(
        building_id=str(building.id),
        rooms=room_schemas,
        waypoints=waypoint_schemas
    )

@router.post("/{building_id}/nav-graph")
async def save_building_graph(
    building_id: str,
    graph_data: NavigationGraphRequest,
    current_user: User = Depends(get_admin_user)
):
    """Save the navigation graph (nodes and edges) for a building"""
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    building = await Building.get(ObjectId(building_id))
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
        
    # Check if graph exists
    graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
    
    if graph:
        graph.nodes = graph_data.nodes
        graph.last_updated = datetime.utcnow()
        await graph.save()
    else:
        graph = BuildingGraph(
            building_id=ObjectId(building_id),
            nodes=graph_data.nodes
        )
        await graph.insert()
        
    return {"message": "Navigation graph saved successfully", "nodes_count": len(graph_data.nodes)}

@router.get("/{building_id}/nav-graph")
async def get_building_graph(building_id: str):
    """Get the navigation graph for a building"""
    try:
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        building_obj_id = ObjectId(building_id)
        graph = await BuildingGraph.find_one(BuildingGraph.building_id == building_obj_id)
        
        if not graph:
            # Return empty graph if none exists
            return {"nodes": []}
        
        # Convert nodes to dict format
        nodes_data = []
        for node in graph.nodes:
            if isinstance(node, dict):
                nodes_data.append(node)
            else:
                # Convert Pydantic model to dict
                nodes_data.append(node.dict() if hasattr(node, 'dict') else node.model_dump())
        
        return {"nodes": nodes_data}
    except Exception as e:
        print(f"Error getting building graph: {e}")
        import traceback
        traceback.print_exc()
        # Return empty graph instead of error
        return {"nodes": []}
