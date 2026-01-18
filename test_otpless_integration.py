"""
Test script for OTPless integration
Run this after starting the backend server to test OTPless endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_send_otp_phone():
    """Test sending OTP to phone number"""
    print("\n=== Testing Send OTP (Phone) ===")
    
    payload = {
        "channel": "PHONE",
        "phone": "+919876543210"
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/otpless/send-otp",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_send_otp_email():
    """Test sending OTP to email"""
    print("\n=== Testing Send OTP (Email) ===")
    
    payload = {
        "channel": "EMAIL",
        "email": "test@example.com"
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/otpless/send-otp",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_verify_register():
    """Test OTPless registration"""
    print("\n=== Testing OTPless Registration ===")
    
    # First send OTP
    otp_response = test_send_otp_phone()
    
    if not otp_response.get("success"):
        print("Failed to send OTP, skipping registration test")
        return
    
    # Get OTP from user
    otp = input("\nEnter the OTP received: ")
    
    payload = {
        "channel": "PHONE",
        "phone": "+919876543210",
        "otp": otp,
        "username": "test_otpless_user",
        "role": "user"
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/otpless/verify-register",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_verify_login():
    """Test OTPless login"""
    print("\n=== Testing OTPless Login ===")
    
    # First send OTP
    otp_response = test_send_otp_phone()
    
    if not otp_response.get("success"):
        print("Failed to send OTP, skipping login test")
        return
    
    # Get OTP from user
    otp = input("\nEnter the OTP received: ")
    
    payload = {
        "channel": "PHONE",
        "phone": "+919876543210",
        "otp": otp
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/otpless/verify-login",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_traditional_register():
    """Test traditional username/password registration"""
    print("\n=== Testing Traditional Registration ===")
    
    payload = {
        "username": "test_traditional_user",
        "email": "traditional@example.com",
        "password": "securepassword123",
        "role": "user"
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_traditional_login():
    """Test traditional username/password login"""
    print("\n=== Testing Traditional Login ===")
    
    payload = {
        "username": "test_traditional_user",
        "password": "securepassword123"
    }
    
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_get_current_user(token):
    """Test getting current user with token"""
    print("\n=== Testing Get Current User ===")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(
        f"{BASE_URL}/auth/me",
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def main():
    """Run all tests"""
    print("=" * 60)
    print("OTPless Integration Test Suite")
    print("=" * 60)
    print("\nMake sure the backend server is running on http://localhost:8000")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Test menu
    while True:
        print("\n" + "=" * 60)
        print("Select a test to run:")
        print("1. Send OTP (Phone)")
        print("2. Send OTP (Email)")
        print("3. OTPless Registration (Phone)")
        print("4. OTPless Login (Phone)")
        print("5. Traditional Registration")
        print("6. Traditional Login")
        print("7. Get Current User (requires token)")
        print("8. Run All Tests")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice: ")
        
        if choice == "1":
            test_send_otp_phone()
        elif choice == "2":
            test_send_otp_email()
        elif choice == "3":
            result = test_verify_register()
            if result and "access_token" in result:
                print(f"\n✓ Registration successful! Token: {result['access_token'][:50]}...")
        elif choice == "4":
            result = test_verify_login()
            if result and "access_token" in result:
                print(f"\n✓ Login successful! Token: {result['access_token'][:50]}...")
        elif choice == "5":
            result = test_traditional_register()
            if result and "access_token" in result:
                print(f"\n✓ Registration successful! Token: {result['access_token'][:50]}...")
        elif choice == "6":
            result = test_traditional_login()
            if result and "access_token" in result:
                print(f"\n✓ Login successful! Token: {result['access_token'][:50]}...")
        elif choice == "7":
            token = input("Enter JWT token: ")
            test_get_current_user(token)
        elif choice == "8":
            print("\n" + "=" * 60)
            print("Running all tests...")
            print("=" * 60)
            test_send_otp_phone()
            test_send_otp_email()
            print("\nNote: Registration and login tests require manual OTP entry")
        elif choice == "0":
            print("\nExiting...")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
