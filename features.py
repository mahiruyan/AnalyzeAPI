from typing import Dict, Any, List
from pathlib import Path

import numpy as np

# Opsiyonel bağımlılıklar: bulunamazsa özellikler kısıtlı hesaplanır
try:
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

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

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
    try:
        data, rate = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        return float(loudness)
    except Exception as e:
        print(f"⚠️ [LOUDNESS] Error: {e}")
        return -20.0  # Default LUFS


def _compute_tempo(audio_path: str) -> float:
    if librosa is None:
        return 0.0
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception as e:
        print(f"⚠️ [TEMPO] Error: {e}")
        return 120.0  # Default BPM


def _ocr_frames(frames: List[str], fast_mode: bool = False) -> List[str]:
    """
    Frame'lerden OCR ile metin çıkarır - GEÇİCİ OLARAK DEVRE DIŞI
    """
    # GEÇİCİ: OCR'ı devre dışı bırak, analiz çalışsın
    print("⚠️ [OCR] Temporarily disabled - returning empty list")
    return []


def _fallback_text_detection(frames: List[str]) -> List[str]:
    """
    GEÇİCİ OLARAK DEVRE DIŞI
    """
    print("⚠️ [OCR] Fallback disabled")
    return []


def _optical_flow(frames: List[str]) -> float:
    if cv2 is None or len(frames) < 2:
        return 0.0
    magnitudes = []
    prev = cv2.imread(frames[0])
    if prev is None:
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Hızlı analiz için sadece ilk 8 kare
    for f in frames[1:8]:  # 15'ten 8'e düşürdük
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
    # Sahne sayısı tahmini için frame dosyalarından video yoluna geri dönmek zor olabilir; bu nedenle frames listesinden klasörü bulup video path'ini çıkaramıyoruz.
    # Bu yüzden scene detect'i atlıyoruz veya kullanıcı video yolunu features'a ekleyebilir. Şimdilik 0 döndürelim.
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


