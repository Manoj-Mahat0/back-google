from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
import json
from bson import ObjectId

from models import Building, Floor, Room, Waypoint, User, BuildingGraph
from schemas import CoordinateGPS
from auth_utils import get_admin_user

router = APIRouter()

@router.post("/buildings/{building_id}/generate-3d")
async def generate_3d_model(
    building_id: str,
    coordinates: List[CoordinateGPS],
    current_user: User = Depends(get_admin_user)
):
    """Generate 3D model from GPS coordinates - now uses navigation graph data"""
    
    try:
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        building = await Building.get(ObjectId(building_id))
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        print(f"🎯 Generating 3D model for building: {building.name}")
        
        # First, check if navigation graph exists (from coordinate collection)
        graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
        
        if graph and graph.nodes:
            print(f"📊 Using existing navigation graph with {len(graph.nodes)} nodes")
            # Use existing navigation graph data
            processed_data = await process_navigation_graph_to_3d(graph, building_id)
        else:
            print(f"📍 Creating new navigation graph from {len(coordinates)} coordinates")
            # Create navigation graph from coordinates first, then use it
            await create_navigation_graph_from_coordinates(coordinates, building_id)
            
            # Now get the created graph
            graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
            processed_data = await process_navigation_graph_to_3d(graph, building_id)
        
        # Create floors, rooms, and waypoints from navigation graph
        result = await create_building_structure_from_graph(processed_data, building_id)
        
        return {
            "message": "3D model generated successfully using navigation graph data",
            "source": "navigation_graph" if graph else "coordinates",
            "nodes_processed": len(graph.nodes) if graph else len(coordinates),
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Error in generate_3d_model: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating 3D model: {str(e)}")

async def create_navigation_graph_from_coordinates(coordinates: List[CoordinateGPS], building_id: str):
    """Create navigation graph from coordinates (same logic as coordinate collection page)"""
    
    # Generate unique timestamp for this batch
    from datetime import datetime
    import time
    batch_id = int(time.time() * 1000)  # Milliseconds since epoch
    
    # Convert coordinates to navigation nodes
    nodes = []
    for index, coord in enumerate(coordinates):
        node = {
            "id": f"node_{batch_id}_{index}",
            "label": coord.room_name if hasattr(coord, 'room_name') and coord.room_name else f"Node {index + 1}",
            "x": coord.latitude,
            "y": coord.longitude,
            "z": float(coord.floor_number),
            "node_type": _determine_node_type(coord.room_name if hasattr(coord, 'room_name') else ""),
            "image_url": None,
            "neighbors": _calculate_neighbors(index, len(coordinates), batch_id),
        }
        nodes.append(node)
    
    # Save navigation graph
    graph = BuildingGraph(
        building_id=ObjectId(building_id),
        nodes=nodes
    )
    await graph.insert()
    print(f"✅ Created navigation graph with {len(nodes)} nodes")

def _determine_node_type(room_name: str) -> str:
    """Determine node type from room name"""
    if not room_name:
        return "waypoint"
    
    name = room_name.lower()
    if 'entrance' in name or 'entry' in name:
        return 'entrance'
    if 'exit' in name:
        return 'exit'
    if 'elevator' in name or 'lift' in name:
        return 'elevator'
    if 'stairs' in name or 'stair' in name:
        return 'stairs'
    if 'bathroom' in name or 'restroom' in name or 'toilet' in name:
        return 'bathroom'
    if 'office' in name:
        return 'office'
    if 'room' in name:
        return 'room'
    if 'lobby' in name or 'reception' in name:
        return 'lobby'
    if 'corridor' in name or 'hallway' in name:
        return 'corridor'
    return 'waypoint'

def _calculate_neighbors(current_index: int, total_nodes: int, batch_id: int) -> List[str]:
    """Calculate neighbors for navigation graph"""
    neighbors = []
    
    # Connect to previous node
    if current_index > 0:
        neighbors.append(f"node_{batch_id}_{current_index - 1}")
    
    # Connect to next node
    if current_index < total_nodes - 1:
        neighbors.append(f"node_{batch_id}_{current_index + 1}")
    
    return neighbors

async def process_navigation_graph_to_3d(graph: BuildingGraph, building_id: str) -> Dict:
    """Convert navigation graph to 3D model data"""
    
    processed_data = {
        'floors': [],
        'rooms': [],
        'waypoints': []
    }
    
    # Group nodes by floor
    floors_dict = {}
    for node in graph.nodes:
        node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
        floor_num = int(node_data.get('z', 0))
        
        if floor_num not in floors_dict:
            floors_dict[floor_num] = []
        floors_dict[floor_num].append(node_data)
    
    # Process each floor
    for floor_num, nodes in floors_dict.items():
        # Add floor info
        floor_info = {
            'number': floor_num,
            'height': 3.0,
            'node_count': len(nodes)
        }
        processed_data['floors'].append(floor_info)
        
        # Convert nodes to rooms and waypoints
        for node in nodes:
            node_type = node.get('node_type', 'waypoint')
            
            # Create room for office/room type nodes
            if node_type in ['office', 'room', 'lobby']:
                room = {
                    'name': node.get('label', 'Unknown Room'),
                    'type': node_type,
                    'coordinates': {
                        'lat': float(node.get('x', 0)),
                        'lng': float(node.get('y', 0)),
                        'floor': floor_num,
                        'width': 5.0,  # Default room size
                        'length': 4.0
                    },
                    'node_id': node.get('id'),
                    'floor_number': floor_num
                }
                processed_data['rooms'].append(room)
            
            # Create waypoint for all nodes
            waypoint = {
                'latitude': float(node.get('x', 0)),
                'longitude': float(node.get('y', 0)),
                'floor_number': floor_num,
                'type': node_type,
                'name': node.get('label', 'Unknown Point'),
                'room_name': node.get('label'),
                'notes': f"Generated from navigation node {node.get('id')}",
                'images': [node.get('image_url')] if node.get('image_url') else None,
                'node_id': node.get('id')
            }
            processed_data['waypoints'].append(waypoint)
    
    print(f"📊 Processed navigation graph: {len(processed_data['floors'])} floors, {len(processed_data['rooms'])} rooms, {len(processed_data['waypoints'])} waypoints")
    return processed_data

async def create_building_structure_from_graph(processed_data: Dict, building_id: str) -> Dict:
    """Create building structure (floors, rooms, waypoints) from processed navigation graph data"""
    
    # Create floors if they don't exist
    floor_data = processed_data.get('floors', [])
    floor_db_ids = {}
    
    for floor_info in floor_data:
        existing_floor = await Floor.find_one(
            Floor.building_id == ObjectId(building_id),
            Floor.floor_number == floor_info['number']
        )
        
        if not existing_floor:
            new_floor = Floor(
                building_id=ObjectId(building_id),
                floor_number=floor_info['number'],
                name=f"Floor {floor_info['number']}",
                height=floor_info.get('height', 3.0)
            )
            await new_floor.insert()
            floor_db_ids[floor_info['number']] = new_floor.id
            print(f"✅ Created floor {floor_info['number']}")
        else:
            floor_db_ids[floor_info['number']] = existing_floor.id
            print(f"📋 Using existing floor {floor_info['number']}")
    
    rooms_created = 0
    waypoints_created = 0
    
    # Create rooms from navigation graph data
    for room_info in processed_data.get('rooms', []):
        floor_num = room_info['coordinates'].get('floor', 0)
        floor_id = floor_db_ids.get(floor_num)
        if floor_id:
            # Check if room already exists
            existing_room = await Room.find_one(
                Room.floor_id == floor_id,
                Room.name == room_info['name']
            )
            
            if not existing_room:
                new_room = Room(
                    floor_id=floor_id,
                    name=room_info['name'],
                    room_type=room_info['type'],
                    coordinates=room_info['coordinates']
                )
                await new_room.insert()
                rooms_created += 1
                print(f"✅ Created room: {room_info['name']}")
    
    # Create waypoints from navigation graph data
    for waypoint_info in processed_data.get('waypoints', []):
        floor_num = waypoint_info['floor_number']
        floor_id = floor_db_ids.get(floor_num)
        if floor_id:
            # Check if waypoint already exists
            existing_waypoint = await Waypoint.find_one(
                Waypoint.floor_id == floor_id,
                Waypoint.name == waypoint_info['name']
            )
            
            if not existing_waypoint:
                new_waypoint = Waypoint(
                    floor_id=floor_id,
                    latitude=waypoint_info['latitude'],
                    longitude=waypoint_info['longitude'],
                    floor_number=waypoint_info['floor_number'],
                    waypoint_type=waypoint_info['type'],
                    name=waypoint_info['name'],
                    room_name=waypoint_info.get('room_name'),
                    notes=waypoint_info.get('notes'),
                    images=waypoint_info.get('images')
                )
                await new_waypoint.insert()
                waypoints_created += 1
                print(f"✅ Created waypoint: {waypoint_info['name']}")
    
    return {
        "floors_processed": len(floor_data),
        "rooms_created": rooms_created,
        "waypoints_created": waypoints_created,
        "model_data": processed_data
    }

@router.get("/analytics/buildings")
async def get_building_analytics(
    current_user: User = Depends(get_admin_user)
):
    """Get analytics data for admin dashboard"""
    
    try:
        total_buildings = await Building.count()
        total_floors = await Floor.count()
        total_rooms = await Room.count()
        total_waypoints = await Waypoint.count()
        
        buildings = await Building.find_all().to_list()
        
        # Convert buildings to dict format for JSON serialization
        buildings_data = []
        for building in buildings:
            building_dict = {
                "id": str(building.id),
                "name": building.name,
                "description": building.description,
                "address": building.address,
                "latitude": building.latitude,
                "longitude": building.longitude,
                "boundary_points": building.boundary_points,
                "created_by": str(building.created_by),
                "created_at": building.created_at.isoformat() if building.created_at else None
            }
            buildings_data.append(building_dict)
        
        return {
            "total_buildings": total_buildings,
            "total_floors": total_floors,
            "total_rooms": total_rooms,
            "total_waypoints": total_waypoints,
            "buildings": buildings_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")