#!/usr/bin/env python3
"""
Simple API test
"""

import requests
import json
import time

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("API is running!")
            return True
        else:
            print("API not responding properly")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_simple_analyze():
    """Test simple analyze request"""
    url = "http://localhost:8000/api/analyze"
    
    # Simple test data
    data = {
        "platform": "instagram",
        "file_url": "https://www.instagram.com/reels/DLEeaIpiryX/",
        "mode": "FAST"
    }
    
    try:
        print("Sending analyze request...")
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis successful!")
            print(f"Score: {result.get('score', 'N/A')}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def main():
    """Main test"""
    print("Simple API Test")
    print("=" * 20)
    
    # Wait for server
    time.sleep(3)
    
    # Test health
    if not test_health():
        print("Server not running, starting server...")
        return False
    
    # Test analyze
    print("\nTesting analyze endpoint...")
    return test_simple_analyze()

if __name__ == "__main__":
    main()
