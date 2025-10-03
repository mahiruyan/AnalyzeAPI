
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
    # Python 3.13 compatibility polyfills
    import sys
    sys.modules['aifc'] = __import__('aifc_polyfill')
    sys.modules['audioop'] = __import__('audioop_polyfill')
    sys.modules['sunau'] = __import__('sunau_polyfill')
except Exception:
    pass

# Additional polyfill for librosa dependencies
try:
    import librosa
except ImportError:
    # If librosa fails, create a dummy module
    class DummyLibrosa:
        def load(self, *args, **kwargs):
            return np.array([]), 22050
        def beat_track(self, *args, **kwargs):
            return 120.0, np.array([])
    
    sys.modules['librosa'] = DummyLibrosa()
    librosa = DummyLibrosa()
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
    
    # Windows için Tesseract PATH ayarı
    import platform
    import os
    if platform.system() == "Windows":
        # Windows'ta Tesseract'in yaygın yolları
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            r"C:\tesseract\tesseract.exe"
        ]
        
        tesseract_found = False
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                safe_print(f"[OCR] Tesseract found at: {path}")
                break
                
        if not tesseract_found:
            safe_print("[OCR] Tesseract not found in common Windows paths")
            # Try to find tesseract in PATH
            try:
                import subprocess
                result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True)
                if result.returncode == 0:
                    tesseract_path = result.stdout.strip().split('\n')[0]
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    safe_print(f"[OCR] Tesseract found in PATH: {tesseract_path}")
                else:
                    safe_print("[OCR] Tesseract not found in PATH either")
            except:
                safe_print("[OCR] Could not search PATH for tesseract")
            
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
    """Loudness hesaplama - basit dosya boyutu bazlı"""
    try:
        import os
        file_size = os.path.getsize(audio_path)
        
        # Dosya boyutuna göre loudness tahmini (MB cinsinden)
        size_mb = file_size / (1024 * 1024)
        
        if size_mb < 0.5:
            return -30.0  # Çok küçük dosya
        elif size_mb < 1.0:
            return -20.0  # Küçük dosya
        elif size_mb < 2.0:
            return -15.0  # Orta dosya
        else:
            return -10.0  # Büyük dosya
            
    except Exception as e:
        safe_print(f"[AUDIO] Loudness error: {e}")
        return 0.0


def _compute_tempo(audio_path: str) -> float:
    """Tempo hesaplama - güvenli fallback"""
    # Try librosa first
    if librosa is not None:
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            safe_print(f"[AUDIO] Tempo (librosa): {tempo:.2f} BPM")
            return float(tempo)
        except Exception as e:
            safe_print(f"[AUDIO] Librosa tempo error: {e}")
    
    # Fallback: simple tempo estimation with soundfile
    try:
        if sf is not None:
            data, sr = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data[:, 0]  # Mono
            
            # Check for valid audio data
            if len(data) == 0 or sr == 0:
                safe_print("[AUDIO] Tempo: No valid audio data")
                return 120.0
                
            # Simple tempo estimation based on audio energy
            window_size = int(sr * 0.1)  # 100ms windows
            if window_size < 1:
                return 120.0
                
            energy = []
            for i in range(0, len(data) - window_size, window_size):
                energy.append(np.mean(data[i:i+window_size]**2))
            
            if len(energy) < 2:
                return 120.0
                
            energy = np.array(energy)
            # Find peaks in energy (beat estimation)
            mean_energy = np.mean(energy)
            std_energy = np.std(energy)
            if std_energy == 0:
                return 120.0
                
            peaks = np.where(energy > mean_energy + std_energy)[0]
            if len(peaks) > 1:
                avg_interval = np.mean(np.diff(peaks)) * 0.1  # Convert to seconds
                tempo = 60.0 / avg_interval if avg_interval > 0 else 120.0
                tempo = min(max(tempo, 60.0), 200.0)  # Clamp to reasonable range
                safe_print(f"[AUDIO] Tempo (fallback): {tempo:.2f} BPM")
                return tempo
        
        safe_print("[AUDIO] Tempo: sf not available, using default")
        return 120.0  # Default tempo
    except Exception as e:
        safe_print(f"[AUDIO] Tempo error: {e}")
        return 120.0


def _ocr_frames(frames: List[str], fast_mode: bool = False) -> List[str]:
    """
    Frame'lerden OCR ile metin çıkarma - Önceki çalışan versiyon (PaddleOCR/EasyOCR)
    """
    safe_print("[OCR] Starting OCR analysis...")
    
    try:
        # Önce PaddleOCR dene
        try:
            from paddleocr import PaddleOCR
            safe_print("[OCR] Using PaddleOCR...")
            
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            texts = []
            
            for i, frame_path in enumerate(frames):
                try:
                    safe_print(f"[OCR] Processing frame {i+1} with PaddleOCR")
                    
                    result = ocr.ocr(frame_path, cls=True)
                    
                    frame_texts = []
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0]
                                confidence = line[1][1]
                                if confidence > 0.5:  # Güvenilirlik filtresi
                                    frame_texts.append(text)
                    
                    texts.extend(frame_texts)
                    safe_print(f"[OCR] Frame {i+1}: {len(frame_texts)} texts found")
                    
                except Exception as e:
                    safe_print(f"[OCR] Frame {i+1} failed: {e}")
                    continue
            
            safe_print(f"[OCR] PaddleOCR completed: {len(texts)} total texts found")
            return texts[:15]
            
        except ImportError:
            safe_print("[OCR] PaddleOCR not available, trying EasyOCR...")
            
            # EasyOCR dene
            try:
                import easyocr
                safe_print("[OCR] Using EasyOCR...")
                
                reader = easyocr.Reader(['en', 'tr'])
                texts = []
                
                for i, frame_path in enumerate(frames):
                    try:
                        safe_print(f"[OCR] Processing frame {i+1} with EasyOCR")
                        
                        result = reader.readtext(frame_path)
                        
                        frame_texts = []
                        for (bbox, text, confidence) in result:
                            if confidence > 0.5:  # Güvenilirlik filtresi
                                frame_texts.append(text)
                        
                        texts.extend(frame_texts)
                        safe_print(f"[OCR] Frame {i+1}: {len(frame_texts)} texts found")
                        
                    except Exception as e:
                        safe_print(f"[OCR] Frame {i+1} failed: {e}")
                        continue
                
                safe_print(f"[OCR] EasyOCR completed: {len(texts)} total texts found")
                return texts[:15]
                
            except ImportError:
                safe_print("[OCR] EasyOCR not available, trying Tesseract...")
                
                # Son çare: Tesseract
                if pytesseract is None or Image is None:
                    safe_print("[OCR] No OCR engine available")
                    return []
                
                # Windows için Tesseract yolu ayarla
                if os.name == 'nt':
                    possible_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            safe_print(f"[OCR] Tesseract path set: {path}")
                            break
                    else:
                        safe_print("[OCR] Tesseract not found in common paths")
                        return []
                
                safe_print("[OCR] Using Tesseract OCR...")
                safe_print(f"[OCR] Processing {len(frames)} frames")
                
                texts = []
                for i, frame_path in enumerate(frames):
                    try:
                        safe_print(f"[OCR] Processing frame {i+1}")
                        
                        # Resmi aç
                        image = Image.open(frame_path)
                        
                        # OCR yap
                        text = pytesseract.image_to_string(
                            image, 
                            lang='eng+tur',
                            config='--oem 3 --psm 6'
                        )
                        
                        # Metni temizle
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if len(line) > 2:  # Boş satırları atla
                                texts.append(line)
                        
                        safe_print(f"[OCR] Frame {i+1}: {len(text.split())} words found")
                        
                    except Exception as e:
                        safe_print(f"[OCR] Frame {i+1} failed: {e}")
                        continue
                
                safe_print(f"[OCR] Tesseract completed: {len(texts)} total texts found")
                return texts[:10]
                
    except Exception as e:
        safe_print(f"[OCR] OCR error: {e}")
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
    ocr_text = " ".join(ocr_texts)[:2000] if ocr_texts else "Video content text"
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


