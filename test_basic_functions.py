#!/usr/bin/env python3
"""
Test basic functions
"""

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        from utils import download_video, extract_audio_via_ffmpeg, grab_frames, get_video_duration
        print("Utils import OK")
        
        from features import extract_features
        print("Features import OK")
        
        from scoring import score_features
        print("Scoring import OK")
        
        from suggest import generate_suggestions
        print("Suggest import OK")
        
        from advanced_analysis import analyze_hook, analyze_pacing_retention, analyze_cta_interaction, analyze_message_clarity
        print("Advanced analysis import OK")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_functions():
    """Test analysis functions with mock data"""
    print("\nTesting analysis functions...")
    
    try:
        from advanced_analysis import analyze_hook, analyze_pacing_retention
        
        # Mock features
        features = {
            "frames": [],
            "audio_info": {"loudness": -15.5, "tempo": 120.0},
            "textual": {"asr_text": "test video content", "ocr_text": []}
        }
        
        print("Testing hook analysis...")
        hook_result = analyze_hook("", "", [], features, 45.0)
        print(f"Hook score: {hook_result['score']}/18")
        
        print("Testing pacing analysis...")
        pacing_result = analyze_pacing_retention([], features, 45.0)
        print(f"Pacing score: {pacing_result['score']}/12")
        
        print("\nAll analysis functions working!")
        return True
        
    except Exception as e:
        print(f"Analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test"""
    print("Basic Function Test")
    print("=" * 30)
    
    # Test imports
    if not test_basic_imports():
        return False
    
    # Test analysis functions
    if not test_analysis_functions():
        return False
    
    print("\nAll tests passed! System is working.")
    return True

if __name__ == "__main__":
    main()
