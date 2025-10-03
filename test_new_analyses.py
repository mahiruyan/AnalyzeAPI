#!/usr/bin/env python3
"""
Test new analyses
"""

def test_new_analyses():
    """Test music sync and accessibility analyses"""
    print("Testing new analyses...")
    
    try:
        from advanced_analysis import analyze_music_sync, analyze_accessibility
        
        # Mock features
        features = {
            "frames": [],
            "audio_info": {"loudness": -15.5, "tempo": 120.0, "spectral_centroid": 2000.0},
            "textual": {"asr_text": "moda güzellik test müzik", "ocr_text": ["MÜZİK", "SENKRON"]}
        }
        
        print("Testing music sync analysis...")
        music_result = analyze_music_sync(features, 45.0)
        print(f"Music sync score: {music_result['score']}/5")
        print(f"Findings: {music_result['findings']}")
        
        print("\nTesting accessibility analysis...")
        access_result = analyze_accessibility(features, 45.0)
        print(f"Accessibility score: {access_result['score']}/4")
        print(f"Findings: {access_result['findings']}")
        
        print("\nAll new analyses passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_new_analyses()
