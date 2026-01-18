"""
Test script for AI Assistant functionality
Run this to verify Groq API integration works
"""
import asyncio
from services.groq_service import groq_service


async def test_label_suggestions():
    """Test smart label generation"""
    print("=" * 60)
    print("Testing Label Suggestions")
    print("=" * 60)
    
    test_cases = [
        {"node_type": "room", "number": "101", "name": None},
        {"node_type": "room", "number": "205", "name": None},
        {"node_type": "entrance", "number": None, "name": None},
        {"node_type": "cafe", "number": None, "name": "Starbucks"},
        {"node_type": "office", "number": "305", "name": None},
        {"node_type": "elevator", "number": "A", "name": None},
        {"node_type": "bathroom", "number": "2", "name": None},
    ]
    
    for case in test_cases:
        label = groq_service.suggest_label(**case)
        print(f"\nInput: {case}")
        print(f"Output: {label}")


async def test_description_generation():
    """Test AI-powered description generation"""
    print("\n" + "=" * 60)
    print("Testing AI Description Generation")
    print("=" * 60)
    
    test_cases = [
        {
            "node_type": "room",
            "label": "Room 301",
            "context": "Computer Science Department"
        },
        {
            "node_type": "entrance",
            "label": "Main Entrance",
            "context": "Ground floor, facing parking lot"
        },
        {
            "node_type": "cafe",
            "label": "Café Starbucks",
            "context": "First floor, near library"
        },
        {
            "node_type": "elevator",
            "label": "Elevator A",
            "context": "Main lobby area"
        },
    ]
    
    for case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Input:")
        print(f"  Type: {case['node_type']}")
        print(f"  Label: {case['label']}")
        print(f"  Context: {case['context']}")
        
        description = groq_service.generate_landmark_description(**case)
        
        print(f"\nGenerated Description:")
        print(f"  {description}")
        print(f"  Word count: {len(description.split())}")


async def main():
    """Run all tests"""
    print("\n🤖 AI Assistant Test Suite\n")
    
    # Test label suggestions
    await test_label_suggestions()
    
    # Test description generation
    await test_description_generation()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
