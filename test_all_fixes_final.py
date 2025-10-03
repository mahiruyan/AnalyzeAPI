#!/usr/bin/env python3

import requests
import json

# Test all fixes
url = "http://localhost:8000/api/analyze"

data = {
    "platform": "instagram",
    "file_url": "https://www.instagram.com/reels/DLEeaIpiryX/",
    "mode": "FULL",
    "title": "Final Fixes Test",
    "caption": "Testing all fixes",
    "tags": ["test", "fixes"]
}

print("Testing all fixes - Final version...")
try:
    response = requests.post(url, json=data, timeout=60)
    if response.status_code == 200:
        result = response.json()
        print(f"API Success: {response.status_code}")
        
        # Check audio features
        audio_features = result.get('features', {}).get('audio', {})
        print(f"Audio loudness: {audio_features.get('loudness_lufs', 'N/A')}")
        print(f"Audio tempo: {audio_features.get('tempo_bpm', 'N/A')}")
        
        # Check OCR
        ocr_text = result.get('features', {}).get('textual', {}).get('ocr_text', [])
        print(f"OCR texts found: {len(ocr_text) if isinstance(ocr_text, list) else 'N/A'}")
        
        # Check if fixes worked
        fixes_status = {
            "Sunau module": "FIXED" if "No module named 'sunau'" not in str(result) else "FAILED",
            "Loudness": "FIXED" if audio_features.get('loudness_lufs', 0.0) != 0.0 else "FAILED",
            "Tempo": "FIXED" if audio_features.get('tempo_bpm', 0.0) != 0.0 else "FAILED",
            "OCR": "FIXED" if isinstance(ocr_text, list) and len(ocr_text) > 0 else "FAILED",
            "OpenCV": "FIXED" if "calcOpticalFlowPyrLK" not in str(result) else "FAILED"
        }
        
        print("\n=== FINAL FIXES STATUS ===")
        for fix, status in fixes_status.items():
            print(f"{fix}: {status}")
            
        print(f"\nOverall score: {result.get('scores', {}).get('overall_score', 'N/A')}")
        
        # Count successful fixes
        successful = sum(1 for status in fixes_status.values() if status == "FIXED")
        total = len(fixes_status)
        print(f"\nSuccess rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
    else:
        print(f"API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Test failed: {e}")
