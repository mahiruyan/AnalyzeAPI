# sunau_polyfill.py
# Comprehensive polyfill for the 'sunau' module, which was removed in Python 3.13.
# This provides full compatibility for librosa and other audio libraries.

class Error(Exception):
    pass

class Au_read:
    def __init__(self, *args, **kwargs):
        self._file = None
        self._nchannels = 1
        self._sampwidth = 2
        self._framerate = 44100
        self._nframes = 0
        self._comptype = 'NONE'
        self._compname = 'not compressed'
        
    def close(self):
        if self._file:
            self._file.close()
            
    def getnchannels(self):
        return self._nchannels
        
    def getsampwidth(self):
        return self._sampwidth
        
    def getframerate(self):
        return self._framerate
        
    def getnframes(self):
        return self._nframes
        
    def getcomptype(self):
        return self._comptype
        
    def getcompname(self):
        return self._compname
        
    def readframes(self, nframes):
        # Return empty bytes as fallback
        return b''
        
    def setpos(self, pos):
        pass
        
    def tell(self):
        return 0

class Au_write:
    def __init__(self, *args, **kwargs):
        self._file = None
        self._nchannels = 1
        self._sampwidth = 2
        self._framerate = 44100
        self._nframes = 0
        self._comptype = 'NONE'
        self._compname = 'not compressed'
        
    def close(self):
        if self._file:
            self._file.close()
            
    def setnchannels(self, nchannels):
        self._nchannels = nchannels
        
    def setsampwidth(self, sampwidth):
        self._sampwidth = sampwidth
        
    def setframerate(self, framerate):
        self._framerate = framerate
        
    def setnframes(self, nframes):
        self._nframes = nframes
        
    def setcomptype(self, comptype, compname):
        self._comptype = comptype
        self._compname = compname
        
    def writeframes(self, data):
        # No-op for fallback
        pass
        
    def writeframesraw(self, data):
        # No-op for fallback
        pass

def open(*args, **kwargs):
    # Return a dummy reader for compatibility
    return Au_read()

# Audio file format constants
AUDIO_FILE_MAGIC = 0x2e736e64
AUDIO_FILE_ENCODING_LINEAR_8 = 1
AUDIO_FILE_ENCODING_LINEAR_16 = 2
AUDIO_FILE_ENCODING_LINEAR_24 = 3
AUDIO_FILE_ENCODING_LINEAR_32 = 4
AUDIO_FILE_ENCODING_ULAW = 5
AUDIO_FILE_ENCODING_ALAW = 6
AUDIO_FILE_ENCODING_IMA_ADPCM = 7
AUDIO_FILE_ENCODING_G722 = 23
AUDIO_FILE_ENCODING_G723_3 = 24
AUDIO_FILE_ENCODING_G723_5 = 25
AUDIO_FILE_ENCODING_G726 = 26
