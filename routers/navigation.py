from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
import math
from bson import ObjectId

from models import Building, Floor, Room, Waypoint, NavigationPath, ARMarker, BuildingGraph, IndoorGraph
from schemas import NavigationRequest, NavigationResponse
from auth_utils import get_current_user

router = APIRouter()

def calculate_distance(point1: Dict[str, float], point2: Dict[str, float]) -> float:
    """Calculate distance between two GPS points"""
    # Simple distance calculation - in production, use proper GPS distance formula
    lat_diff = point1['latitude'] - point2['latitude']
    lng_diff = point1['longitude'] - point2['longitude']
    floor_diff = (point1.get('floor_number', 0) - point2.get('floor_number', 0)) * 3  # 3m per floor
    
    return math.sqrt(lat_diff**2 + lng_diff**2 + floor_diff**2) * 111000  # Convert to meters approximately

def find_nearest_waypoint(user_pos: Dict[str, float], waypoints: List[Waypoint]) -> Waypoint:
    """Find the nearest waypoint to user position"""
    min_distance = float('inf')
    nearest_waypoint = None
    
    for waypoint in waypoints:
        wp_pos = {
            'latitude': waypoint.latitude, 
            'longitude': waypoint.longitude, 
            'floor_number': waypoint.floor_number
        }
        distance = calculate_distance(user_pos, wp_pos)
        if distance < min_distance:
            min_distance = distance
            nearest_waypoint = waypoint
    
    return nearest_waypoint

def find_path_to_room(start_waypoint: Waypoint, target_room: Room, waypoints: List[Waypoint]) -> List[Dict[str, float]]:
    """Simple pathfinding algorithm"""
    path = []
    
    # Add start waypoint
    path.append({
        'latitude': start_waypoint.latitude, 
        'longitude': start_waypoint.longitude, 
        'floor_number': start_waypoint.floor_number
    })
    
    # Add room center as destination
    room_coords = target_room.coordinates
    room_center = {
        'latitude': room_coords.get('lat', 0) + room_coords.get('width', 0) / 200000,  # Approximate GPS offset
        'longitude': room_coords.get('lng', 0) + room_coords.get('length', 0) / 200000,
        'floor_number': room_coords.get('floor', 0)
    }
    path.append(room_center)
    
    return path


# ============================================
# SMART NAVIGATION ENDPOINTS (No QR Code)
# ============================================

@router.get("/buildings/{building_id}/locations")
async def get_building_locations(building_id: str):
    """Get all navigable locations (waypoints + rooms) for a building"""
    try:
        print(f"🔍 Fetching locations for building: {building_id}")
        
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        building = await Building.get(ObjectId(building_id))
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        locations = []
        
        # Get from IndoorGraph first (preferred - has category support)
        try:
            indoor_graph = await IndoorGraph.find_one(IndoorGraph.building_id == ObjectId(building_id))
            print(f"📊 IndoorGraph found: {indoor_graph is not None}")
            
            if indoor_graph and indoor_graph.nodes:
                print(f"📍 Processing {len(indoor_graph.nodes)} nodes from IndoorGraph")
                
                for i, node in enumerate(indoor_graph.nodes):
                    try:
                        node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
                        location = {
                            "id": node_data.get("id", ""),
                            "name": node_data.get("label") or node_data.get("name") or f"Node {node_data.get('id', '')}",
                            "node_type": node_data.get("node_type", "waypoint"),
                            "floor_number": int(node_data.get("floor_number", 0)),
                            "latitude": float(node_data.get("latitude", 0)),
                            "longitude": float(node_data.get("longitude", 0)),
                            "image_url": node_data.get("image_url"),
                            "landmark_description": node_data.get("landmark_description"),
                            "neighbors": [e.get("to_node_id") for e in node_data.get("edges", [])],
                            "category": node_data.get("category"),
                        }
                        locations.append(location)
                        print(f"✅ Node {i+1}: {location['name']} ({location['node_type']}) - Category: {location.get('category', 'None')}")
                    except Exception as e:
                        print(f"❌ Error processing node {i+1}: {e}")
                        continue
                
                if locations:
                    print(f"🎉 Returning {len(locations)} locations from IndoorGraph")
                    return {"locations": locations}
        except Exception as e:
            print(f"❌ Error accessing IndoorGraph: {e}")
        
        # Fallback to BuildingGraph
        try:
            graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
            print(f"📊 BuildingGraph found: {graph is not None}")
            
            if graph and graph.nodes:
                print(f"📍 Processing {len(graph.nodes)} nodes from BuildingGraph")
                
                for i, node in enumerate(graph.nodes):
                    try:
                        node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
                        location = {
                            "id": node_data.get("id", ""),
                            "name": node_data.get("label") or node_data.get("name") or f"Node {node_data.get('id', '')}",
                            "node_type": node_data.get("node_type", "waypoint"),
                            "floor_number": int(node_data.get("z", 0)),
                            "latitude": float(node_data.get("x", 0)),
                            "longitude": float(node_data.get("y", 0)),
                            "image_url": node_data.get("image_url"),
                            "neighbors": node_data.get("neighbors", []),
                            "category": node_data.get("category"),
                        }
                        locations.append(location)
                        print(f"✅ Node {i+1}: {location['name']} ({location['node_type']})")
                    except Exception as e:
                        print(f"❌ Error processing node {i+1}: {e}")
                        continue
                
                if locations:
                    print(f"🎉 Returning {len(locations)} locations from BuildingGraph")
                    return {"locations": locations}
        except Exception as e:
            print(f"❌ Error accessing BuildingGraph: {e}")
        
        # Fallback to waypoints and rooms
        print("📋 Falling back to waypoints and rooms...")
        try:
            floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
            print(f"🏢 Found {len(floors)} floors")
            
            for floor in floors:
                # Add waypoints
                waypoints = await Waypoint.find(Waypoint.floor_id == floor.id).to_list()
                print(f"📍 Floor {floor.floor_number}: {len(waypoints)} waypoints")
                
                for wp in waypoints:
                    locations.append({
                        "id": str(wp.id),
                        "name": wp.name,
                        "node_type": wp.waypoint_type,
                        "floor_number": wp.floor_number,
                        "latitude": wp.latitude,
                        "longitude": wp.longitude,
                        "image_url": wp.images[0] if wp.images else None,
                        "neighbors": [],
                    })
                
                # Add rooms as destinations
                rooms = await Room.find(Room.floor_id == floor.id).to_list()
                print(f"🏠 Floor {floor.floor_number}: {len(rooms)} rooms")
                
                for room in rooms:
                    coords = room.coordinates
                    locations.append({
                        "id": str(room.id),
                        "name": room.name,
                        "node_type": room.room_type,
                        "floor_number": int(coords.get("floor", floor.floor_number)),
                        "latitude": float(coords.get("lat", 0)),
                        "longitude": float(coords.get("lng", 0)),
                        "image_url": None,
                        "neighbors": [],
                    })
        except Exception as e:
            print(f"❌ Error accessing waypoints/rooms: {e}")
        
        # If no locations found, return sample data for testing
        if not locations:
            print("🔧 No real data found, returning sample locations for testing")
            locations = [
                {
                    "id": "sample_entrance",
                    "name": "Main Entrance",
                    "node_type": "entrance",
                    "floor_number": 0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "image_url": None,
                    "neighbors": ["sample_lobby"],
                },
                {
                    "id": "sample_lobby",
                    "name": "Lobby",
                    "node_type": "lobby",
                    "floor_number": 0,
                    "latitude": 0.001,
                    "longitude": 0.001,
                    "image_url": None,
                    "neighbors": ["sample_entrance", "sample_elevator"],
                },
                {
                    "id": "sample_elevator",
                    "name": "Elevator",
                    "node_type": "elevator",
                    "floor_number": 0,
                    "latitude": 0.002,
                    "longitude": 0.002,
                    "image_url": None,
                    "neighbors": ["sample_lobby"],
                },
            ]
        
        print(f"📤 Final response: {len(locations)} locations")
        return {"locations": locations}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Unexpected error in get_building_locations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/buildings/{building_id}/locations/by-category/{category}")
async def get_locations_by_category(building_id: str, category: str):
    """Get all locations in a building filtered by category for intent-based navigation"""
    try:
        print(f"🔍 Fetching locations by category '{category}' for building: {building_id}")
        
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        building = await Building.get(ObjectId(building_id))
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        locations = []
        
        # Get from navigation graph
        graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
        
        if graph and graph.nodes:
            print(f"📊 Filtering {len(graph.nodes)} nodes by category: {category}")
            
            for node in graph.nodes:
                try:
                    node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
                    node_category = node_data.get("category")
                    
                    # Filter by category
                    if node_category and node_category.lower() == category.lower():
                        location = {
                            "id": node_data.get("id", ""),
                            "name": node_data.get("label") or node_data.get("name") or f"Node {node_data.get('id', '')}",
                            "node_type": node_data.get("node_type", "waypoint"),
                            "floor_number": int(node_data.get("z", 0)),
                            "latitude": float(node_data.get("x", 0)),
                            "longitude": float(node_data.get("y", 0)),
                            "image_url": node_data.get("image_url"),
                            "neighbors": node_data.get("neighbors", []),
                            "category": node_category,
                        }
                        locations.append(location)
                        print(f"✅ Found: {location['name']} (category: {node_category})")
                except Exception as e:
                    print(f"❌ Error processing node: {e}")
                    continue
        
        if not locations:
            print(f"⚠️ No locations found with category '{category}'")
            return {
                "message": f"No locations found with category '{category}'",
                "locations": []
            }
        
        print(f"🎉 Returning {len(locations)} locations with category '{category}'")
        return {
            "category": category,
            "count": len(locations),
            "locations": locations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Unexpected error in get_locations_by_category: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/buildings/{building_id}/categories")
async def get_available_categories(building_id: str):
    """Get all available categories in a building for intent-based navigation"""
    try:
        print(f"🔍 Fetching available categories for building: {building_id}")
        
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        building = await Building.get(ObjectId(building_id))
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        categories = set()
        
        # Get from navigation graph
        graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
        
        if graph and graph.nodes:
            print(f"📊 Extracting categories from {len(graph.nodes)} nodes")
            
            for node in graph.nodes:
                try:
                    node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
                    node_category = node_data.get("category")
                    
                    if node_category:
                        categories.add(node_category)
                except Exception as e:
                    print(f"❌ Error processing node: {e}")
                    continue
        
        categories_list = sorted(list(categories))
        print(f"🎉 Found {len(categories_list)} categories: {categories_list}")
        
        return {
            "building_id": building_id,
            "building_name": building.name,
            "categories": categories_list,
            "count": len(categories_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Unexpected error in get_available_categories: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/buildings/{building_id}/route")
async def calculate_smart_route(building_id: str, route_request: dict):
    """Calculate route between two nodes with step-by-step instructions"""
    try:
        if not ObjectId.is_valid(building_id):
            raise HTTPException(status_code=400, detail="Invalid building ID")
        
        start_node_id = route_request.get("start_node_id")
        end_node_id = route_request.get("end_node_id")
        
        if not start_node_id or not end_node_id:
            raise HTTPException(status_code=400, detail="start_node_id and end_node_id are required")
        
        # Get navigation graph
        graph = await BuildingGraph.find_one(BuildingGraph.building_id == ObjectId(building_id))
        
        if not graph or not graph.nodes:
            # Return sample route for testing
            return {
                "steps": [
                    {
                        "id": start_node_id,
                        "name": "Start Location",
                        "node_type": "waypoint",
                        "floor_number": 0,
                        "latitude": 0.0,
                        "longitude": 0.0,
                        "image_url": None,
                        "instruction": "Start your journey here",
                        "distance": None,
                    },
                    {
                        "id": end_node_id,
                        "name": "Destination",
                        "node_type": "waypoint",
                        "floor_number": 0,
                        "latitude": 0.001,
                        "longitude": 0.001,
                        "image_url": None,
                        "instruction": "You have arrived at your destination",
                        "distance": 10.0,
                    },
                ],
                "total_distance": 10.0,
                "estimated_time": 7,
            }
        
        # Build node lookup
        nodes_dict = {}
        for node in graph.nodes:
            node_data = node if isinstance(node, dict) else (node.dict() if hasattr(node, 'dict') else node.model_dump())
            nodes_dict[node_data.get("id")] = node_data
        
        if start_node_id not in nodes_dict:
            raise HTTPException(status_code=404, detail="Start node not found")
        if end_node_id not in nodes_dict:
            raise HTTPException(status_code=404, detail="End node not found")
        
        # A* pathfinding
        path = _astar_pathfinding(nodes_dict, start_node_id, end_node_id)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found between the selected locations")
        
        # Generate step-by-step instructions
        steps = []
        total_distance = 0
        
        for i, node_id in enumerate(path):
            node = nodes_dict[node_id]
            
            # Calculate distance to next node
            distance = None
            if i < len(path) - 1:
                next_node = nodes_dict[path[i + 1]]
                distance = _calculate_node_distance(node, next_node)
                total_distance += distance
            
            # Generate instruction
            instruction = _generate_instruction(node, nodes_dict.get(path[i + 1]) if i < len(path) - 1 else None, i, len(path))
            
            steps.append({
                "id": node_id,
                "name": node.get("label") or node.get("name") or f"Point {i + 1}",
                "node_type": node.get("node_type", "waypoint"),
                "floor_number": int(node.get("z", 0)),
                "latitude": float(node.get("x", 0)),
                "longitude": float(node.get("y", 0)),
                "image_url": node.get("image_url"),
                "instruction": instruction,
                "distance": distance,
                "category": node.get("category"),
            })
        
        return {
            "steps": steps,
            "total_distance": total_distance,
            "estimated_time": int(total_distance / 1.4) if total_distance > 0 else 0,  # 1.4 m/s walking speed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in calculate_smart_route: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _astar_pathfinding(nodes_dict: dict, start_id: str, end_id: str) -> List[str]:
    """Enhanced A* pathfinding algorithm with inter-floor navigation support"""
    import heapq
    
    def heuristic(node_id: str) -> float:
        node = nodes_dict[node_id]
        end_node = nodes_dict[end_id]
        return _calculate_node_distance(node, end_node)
    
    def can_traverse_floors(current_node: dict, neighbor_node: dict) -> bool:
        """Check if floor transition is possible between nodes"""
        current_floor = int(current_node.get("z", 0))
        neighbor_floor = int(neighbor_node.get("z", 0))
        
        # Same floor - always possible
        if current_floor == neighbor_floor:
            return True
        
        # Different floors - need elevator or stairs
        current_type = current_node.get("node_type", "").lower()
        neighbor_type = neighbor_node.get("node_type", "").lower()
        
        # Check if either node is an elevator or stairs
        floor_transition_types = {"elevator", "stairs", "staircase", "lift", "escalator"}
        
        return (current_type in floor_transition_types or 
                neighbor_type in floor_transition_types)
    
    def find_floor_connectors(start_floor: int, end_floor: int) -> List[str]:
        """Find elevator/stair nodes that can connect floors"""
        connectors = []
        floor_transition_types = {"elevator", "stairs", "staircase", "lift", "escalator"}
        
        for node_id, node in nodes_dict.items():
            node_type = node.get("node_type", "").lower()
            node_floor = int(node.get("z", 0))
            
            if (node_type in floor_transition_types and 
                (node_floor == start_floor or node_floor == end_floor)):
                connectors.append(node_id)
        
        return connectors
    
    # Check if inter-floor navigation is needed
    start_floor = int(nodes_dict[start_id].get("z", 0))
    end_floor = int(nodes_dict[end_id].get("z", 0))
    
    if start_floor != end_floor:
        # Find available floor connectors
        connectors = find_floor_connectors(start_floor, end_floor)
        if not connectors:
            print(f"❌ No floor connectors found between floor {start_floor} and {end_floor}")
            return []  # No inter-floor connections available
    
    open_set = [(0, start_id)]
    came_from = {}
    g_score = {start_id: 0}
    f_score = {start_id: heuristic(start_id)}
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        if current == end_id:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        current_node = nodes_dict[current]
        neighbors = current_node.get("neighbors", [])
        
        for neighbor_id in neighbors:
            if neighbor_id not in nodes_dict or neighbor_id in closed_set:
                continue
            
            neighbor_node = nodes_dict[neighbor_id]
            
            # Check if floor transition is valid
            if not can_traverse_floors(current_node, neighbor_node):
                continue
            
            tentative_g = g_score[current] + _calculate_node_distance(current_node, neighbor_node)
            
            if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                came_from[neighbor_id] = current
                g_score[neighbor_id] = tentative_g
                f_score[neighbor_id] = tentative_g + heuristic(neighbor_id)
                heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
    
    return []  # No path found


def _calculate_node_distance(node1: dict, node2: dict) -> float:
    """Calculate distance between two nodes in meters"""
    x1, y1, z1 = float(node1.get("x", 0)), float(node1.get("y", 0)), float(node1.get("z", 0))
    x2, y2, z2 = float(node2.get("x", 0)), float(node2.get("y", 0)), float(node2.get("z", 0))
    
    # If coordinates are GPS (lat/lng), convert to meters
    lat_diff = (x2 - x1) * 111000  # ~111km per degree latitude
    lng_diff = (y2 - y1) * 111000 * math.cos(math.radians((x1 + x2) / 2))
    floor_diff = (z2 - z1) * 3  # 3m per floor
    
    return math.sqrt(lat_diff**2 + lng_diff**2 + floor_diff**2)


def _generate_instruction(current_node: dict, next_node: dict, step_index: int, total_steps: int) -> str:
    """Generate human-readable navigation instruction"""
    node_name = current_node.get("label") or current_node.get("name") or "this point"
    node_type = current_node.get("node_type", "waypoint")
    
    if step_index == 0:
        return f"Start at {node_name}"
    
    if step_index == total_steps - 1:
        return f"You have arrived at {node_name}"
    
    if not next_node:
        return f"Continue to {node_name}"
    
    # Check for floor change
    current_floor = int(current_node.get("z", 0))
    next_floor = int(next_node.get("z", 0))
    
    if current_floor != next_floor:
        direction = "up" if next_floor > current_floor else "down"
        if node_type in ["elevator", "lift"]:
            return f"Take the elevator {direction} to floor {next_floor}"
        elif node_type in ["stairs", "staircase"]:
            return f"Take the stairs {direction} to floor {next_floor}"
        else:
            return f"Go {direction} to floor {next_floor}"
    
    # Type-specific instructions
    if node_type == "entrance":
        return f"Enter through {node_name}"
    elif node_type == "exit":
        return f"Exit at {node_name}"
    elif node_type == "junction":
        return f"At the junction, continue towards your destination"
    elif node_type in ["bathroom", "restroom"]:
        return f"Pass by {node_name}"
    else:
        return f"Continue to {node_name}"


@router.post("/navigate", response_model=NavigationResponse)
async def get_navigation(
    request: NavigationRequest,
    current_user = Depends(get_current_user)
):
    if not ObjectId.is_valid(str(request.building_id)):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    # Get building
    building = await Building.get(request.building_id)
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    
    # Find target room
    floors = await Floor.find(Floor.building_id == request.building_id).to_list()
    target_room = None
    
    for floor in floors:
        rooms = await Room.find(Room.floor_id == floor.id).to_list()
        for room in rooms:
            if request.destination_room_name.lower() in room.name.lower():
                target_room = room
                break
        if target_room:
            break
    
    if not target_room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Get waypoints for the floor
    waypoints = await Waypoint.find(Waypoint.floor_id == target_room.floor_id).to_list()
    
    if not waypoints:
        raise HTTPException(status_code=404, detail="No waypoints found for this floor")
    
    # Find nearest waypoint to user position
    user_pos = {
        'latitude': request.start_latitude, 
        'longitude': request.start_longitude, 
        'floor_number': request.start_floor
    }
    start_waypoint = find_nearest_waypoint(user_pos, waypoints)
    
    # Generate path
    path = find_path_to_room(start_waypoint, target_room, waypoints)
    
    # Calculate total distance
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += calculate_distance(path[i], path[i + 1])
    
    # Estimate time (assuming 1.4 m/s walking speed)
    estimated_time = int(total_distance / 1.4)
    
    # Generate instructions
    instructions = [
        f"Head to the nearest waypoint",
        f"Navigate to {target_room.name}",
        f"You have arrived at your destination"
    ]
    
    return NavigationResponse(
        path=path,
        distance=total_distance,
        estimated_time=estimated_time,
        instructions=instructions
    )

@router.get("/buildings/{building_id}/ar-markers")
async def get_ar_markers(building_id: str):
    """Get AR markers for a building"""
    if not ObjectId.is_valid(building_id):
        raise HTTPException(status_code=400, detail="Invalid building ID")
    
    # Get all floors for the building
    floors = await Floor.find(Floor.building_id == ObjectId(building_id)).to_list()
    floor_ids = [floor.id for floor in floors]
    
    # Get AR markers for all floors
    markers = await ARMarker.find({"floor_id": {"$in": floor_ids}}).to_list()
    
    return [
        {
            "id": str(marker.id),
            "latitude": marker.latitude,
            "longitude": marker.longitude,
            "floor_number": marker.floor_number,
            "type": marker.marker_type,
            "data": marker.marker_data,
            "description": marker.description
        }
        for marker in markers
    ]