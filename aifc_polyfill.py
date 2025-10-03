"""
aifc polyfill for Python 3.13 compatibility
This module provides a minimal aifc implementation for librosa compatibility
"""

import wave
import struct
import io

class Error(Exception):
    pass

def open(f, mode=None):
    """Minimal aifc.open implementation"""
    if mode == 'rb' or mode is None:
        # Try to read as WAV file first
        try:
            return wave.open(f, 'rb')
        except:
            # If not WAV, return a dummy reader
            return DummyAifcReader(f)
    else:
        return wave.open(f, mode)

class DummyAifcReader:
    """Dummy aifc reader for non-WAV files"""
    def __init__(self, f):
        self.f = f
        self.nframes = 0
        self.sampwidth = 2
        self.nchannels = 1
        self.framerate = 22050
        
    def getnframes(self):
        return self.nframes
        
    def getframerate(self):
        return self.framerate
        
    def readframes(self, nframes):
        # Return silent audio data
        return b'\x00\x00' * nframes
        
    def close(self):
        pass
