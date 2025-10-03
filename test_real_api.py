#!/usr/bin/env python3
"""
Test with real Instagram reel
"""

import requests
import json
import time

def test_instagram_reel():
    """Test with real Instagram reel"""
    print("Testing with real Instagram reel...")
    
    # API endpoint
    url = "http://localhost:8000/api/analyze"
    
    # Test data
    data = {
        "platform": "instagram",
        "file_url": "https://www.instagram.com/reels/DLEeaIpiryX/",
        "mode": "FAST",
        "title": "Test Reel",
        "caption": "Test caption for analysis"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== ANALYSIS RESULTS ===")
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Verdict: {result.get('verdict', 'N/A')}")
            print(f"Viral: {result.get('viral', 'N/A')}")
            print(f"Analysis Complete: {result.get('analysis_complete', 'N/A')}")
            
            # Advanced scores
            scores = result.get('scores', {})
            print("\n=== ADVANCED SCORES ===")
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value}")
            
            # Findings
            findings = result.get('findings', [])
            if findings:
                print(f"\n=== FINDINGS ({len(findings)}) ===")
                for i, finding in enumerate(findings[:5], 1):  # First 5 findings
                    print(f"{i}. {finding}")
            
            # Suggestions
            suggestions = result.get('suggestions', [])
            if suggestions:
                print(f"\n=== SUGGESTIONS ({len(suggestions)}) ===")
                for i, suggestion in enumerate(suggestions[:5], 1):  # First 5 suggestions
                    print(f"{i}. {suggestion}")
            
            return True
        else:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out - API might be processing")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection error - API server not running")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Real API Test with Instagram Reel")
    print("=" * 40)
    
    # Wait a bit for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    success = test_instagram_reel()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
    
    return success

if __name__ == "__main__":
    main()
