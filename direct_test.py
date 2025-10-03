#!/usr/bin/env python3
"""
Direct function test
"""

def test_analysis_functions():
    """Test analysis functions directly"""
    print("Testing analysis functions...")
    
    try:
        from advanced_analysis import analyze_content_type_suitability, analyze_trend_originality
        
        # Mock features
        features = {
            "frames": [],
            "audio_info": {"loudness": -15.5, "tempo": 120.0, "spectral_centroid": 2000.0},
            "textual": {"asr_text": "moda güzellik test", "ocr_text": []}
        }
        
        print("Testing content type analysis...")
        result1 = analyze_content_type_suitability(features, 45.0)
        print(f"Content type score: {result1['score']}/6")
        
        print("Testing trend analysis...")
        result2 = analyze_trend_originality(features, 45.0)
        print(f"Trend score: {result2['score']}/8")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_analysis_functions()
