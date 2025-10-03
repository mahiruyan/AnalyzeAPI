
def safe_print(*args, **kwargs):
    """Safe print function that handles Unicode errors"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic characters and try again
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_arg = arg.encode('ascii', 'replace').decode('ascii')
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)


from typing import Dict, Any, List
from pathlib import Path

import numpy as np

# Opsiyonel bamllklar: bulunamazsa zellikler kstl hesaplanr
try:
    # aifc polyfill for Python 3.13
    import sys
    sys.modules['aifc'] = __import__('aifc_polyfill')
    # audioop polyfill for Python 3.13
    sys.modules['audioop'] = __import__('audioop_polyfill')
    import librosa
except Exception:
    librosa = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

try:
    import pytesseract  # type: ignore
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception:
    VideoManager = None
    SceneManager = None
    ContentDetector = None

# PaddleOCR kaldrld - sadece Tesseract kullanyoruz

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


def _safe_load_audio(audio_path: str, sr: int = 16000):
    if librosa is not None:
        y, _sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr
    if sf is not None:
        y, _sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), _sr
    return np.array([]), sr


def _compute_loudness(audio_path: str) -> float:
    if pyln is None or sf is None:
        return 0.0
    data, rate = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    return float(loudness)


def _compute_tempo(audio_path: str) -> float:
    """Tempo hesaplama - soundfile ile fallback"""
    if librosa is not None:
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except Exception as e:
            safe_print(f"Librosa tempo hatas: {e}")
    
    # Fallback: simple tempo estimation
    try:
        if sf is not None:
            data, sr = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data[:, 0]  # Mono
            # Simple tempo estimation based on audio energy
            window_size = int(sr * 0.1)  # 100ms windows
            energy = []
            for i in range(0, len(data) - window_size, window_size):
                energy.append(np.mean(data[i:i+window_size]**2))
            energy = np.array(energy)
            # Find peaks in energy (beat estimation)
            peaks = np.where(energy > np.mean(energy) + np.std(energy))[0]
            if len(peaks) > 1:
                avg_interval = np.mean(np.diff(peaks)) * 0.1  # Convert to seconds
                tempo = 60.0 / avg_interval if avg_interval > 0 else 120.0
                return min(max(tempo, 60.0), 200.0)  # Clamp to reasonable range
        return 120.0  # Default tempo
    except Exception as e:
        safe_print(f"Fallback tempo hatas: {e}")
        return 120.0


def _ocr_frames(frames: List[str], fast_mode: bool = False) -> List[str]:
    """
    Frame'lerden Tesseract OCR ile metin karr
    """
    texts = []
    
    # OCR her zaman çalışır - fast mode kaldırıldı
    safe_print("[OCR] Starting OCR analysis...")
    
    try:
        if pytesseract is None or Image is None:
            safe_print("[OCR] Tesseract not available")
            return texts
            
        safe_print("[OCR] Using Tesseract OCR...")
        safe_print(f"[OCR] Tesseract initialized, processing {len(frames)} frames")
        
        # Tüm frame'leri analiz et - FULL mode
        for i, f in enumerate(frames):
            try:
                # Grnty a ve OCR uygula
                image = Image.open(f)
                # ngilizce ve Trke iin OCR
                text = pytesseract.image_to_string(image, lang='eng+tur')
                
                # Metni satrlara bl ve temizle
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 1:  # Bo satrlar atla
                        texts.append(line)
                        
                safe_print(f"[OCR] Frame {i+1}: {len([t for t in texts if t])} texts found")
            except Exception as e:
                safe_print(f"[OCR] Failed for frame {i+1}: {e}")
                continue
        
        safe_print(f"[OCR] Completed: {len(texts)} total texts found")
        return texts[:10]  # Maksimum 10 text
        
    except Exception as e:
        safe_print(f"[OCR] Tesseract failed: {e}")
        return []


# Fallback fonksiyonu kaldrld - artk sadece Tesseract kullanyoruz


def _optical_flow(frames: List[str]) -> float:
    if cv2 is None or len(frames) < 2:
        return 0.0
    magnitudes = []
    prev = cv2.imread(frames[0])
    if prev is None:
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Hzl analiz iin sadece ilk 8 kare
    for f in frames[1:8]:  # 15'ten 8'e drdk
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
        prev_gray = gray
    if not magnitudes:
        return 0.0
    return float(np.mean(magnitudes))


def _scene_changes(video_path: str) -> int:
    if VideoManager is None:
        return 0
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        return len(scene_list)
    finally:
        video_manager.release()


def _asr(audio_path: str, fast_mode: bool = False) -> str:
    if WhisperModel is None or fast_mode:
        return ""  # FAST modda ASR atla
    try:
        model = WhisperModel("small", compute_type="int8")
        segments, info = model.transcribe(audio_path, language="en")
        text_parts = [seg.text.strip() for seg in segments]
        return " ".join([p for p in text_parts if p])
    except Exception:
        return ""


def extract_features(
    audio_path: str,
    frames: List[str],
    caption: str,
    title: str,
    tags: List[str],
    duration_seconds: float,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    loudness = _compute_loudness(audio_path)
    tempo = _compute_tempo(audio_path)
    flow = _optical_flow(frames)
    ocr_texts = _ocr_frames(frames, fast_mode)
    ocr_text = " ".join(ocr_texts)[:2000]
    asr_text = _asr(audio_path, fast_mode)
    # Sahne says tahmini iin frame dosyalarndan video yoluna geri dnmek zor olabilir; bu nedenle frames listesinden klasr bulup video path'ini karamyoruz.
    # Bu yzden scene detect'i atlyoruz veya kullanc video yolunu features'a ekleyebilir. imdilik 0 dndrelim.
    scene_count = 0

    return {
        "duration_seconds": float(duration_seconds),
        "audio": {
            "loudness_lufs": loudness,
            "tempo_bpm": tempo,
        },
        "visual": {
            "optical_flow_mean": flow,
            "scene_count": scene_count,
        },
        "textual": {
            "ocr_text": ocr_text,
            "asr_text": asr_text,
            "caption": caption or "",
            "title": title or "",
            "tags": tags or [],
        },
    }


