"""
audioop polyfill for Python 3.13 compatibility
This module provides a minimal audioop implementation for librosa compatibility
"""

import struct

def mul(fragment, width, factor):
    """Multiply audio fragment by factor"""
    if width == 1:
        return b''.join(struct.pack('B', max(0, min(255, int(b * factor)))) for b in fragment)
    elif width == 2:
        return b''.join(struct.pack('<h', max(-32768, min(32767, int(struct.unpack('<h', fragment[i:i+2])[0] * factor)))) for i in range(0, len(fragment), 2))
    elif width == 4:
        return b''.join(struct.pack('<i', max(-2147483648, min(2147483647, int(struct.unpack('<i', fragment[i:i+4])[0] * factor)))) for i in range(0, len(fragment), 4))
    else:
        return fragment

def add(fragment1, fragment2, width):
    """Add two audio fragments"""
    if width == 1:
        return b''.join(struct.pack('B', max(0, min(255, struct.unpack('B', fragment1[i:i+1])[0] + struct.unpack('B', fragment2[i:i+1])[0]))) for i in range(0, min(len(fragment1), len(fragment2))))
    elif width == 2:
        return b''.join(struct.pack('<h', max(-32768, min(32767, struct.unpack('<h', fragment1[i:i+2])[0] + struct.unpack('<h', fragment2[i:i+2])[0]))) for i in range(0, min(len(fragment1), len(fragment2)), 2))
    elif width == 4:
        return b''.join(struct.pack('<i', max(-2147483648, min(2147483647, struct.unpack('<i', fragment1[i:i+4])[0] + struct.unpack('<i', fragment2[i:i+4])[0]))) for i in range(0, min(len(fragment1), len(fragment2)), 4))
    else:
        return fragment1

def bias(fragment, width, bias):
    """Add bias to audio fragment"""
    return add(fragment, struct.pack(f'<{"H" if width == 2 else "I" if width == 4 else "B"}', bias) * (len(fragment) // width), width)

def reverse(fragment, width):
    """Reverse audio fragment"""
    result = bytearray()
    for i in range(0, len(fragment), width):
        result.extend(fragment[i:i+width][::-1])
    return bytes(result)

def tomono(fragment, width, lfactor, rfactor):
    """Convert stereo to mono"""
    if width == 1:
        return b''.join(struct.pack('B', max(0, min(255, int(struct.unpack('B', fragment[i:i+1])[0] * lfactor + struct.unpack('B', fragment[i+1:i+2])[0] * rfactor)))) for i in range(0, len(fragment), 2))
    elif width == 2:
        return b''.join(struct.pack('<h', max(-32768, min(32767, int(struct.unpack('<h', fragment[i:i+2])[0] * lfactor + struct.unpack('<h', fragment[i+2:i+4])[0] * rfactor)))) for i in range(0, len(fragment), 4))
    elif width == 4:
        return b''.join(struct.pack('<i', max(-2147483648, min(2147483647, int(struct.unpack('<i', fragment[i:i+4])[0] * lfactor + struct.unpack('<i', fragment[i+4:i+8])[0] * rfactor)))) for i in range(0, len(fragment), 8))
    else:
        return fragment

def tostereo(fragment, width, lfactor, rfactor):
    """Convert mono to stereo"""
    if width == 1:
        return b''.join(struct.pack('BB', max(0, min(255, int(struct.unpack('B', fragment[i:i+1])[0] * lfactor))), max(0, min(255, int(struct.unpack('B', fragment[i:i+1])[0] * rfactor)))) for i in range(0, len(fragment), 1))
    elif width == 2:
        return b''.join(struct.pack('<hh', max(-32768, min(32767, int(struct.unpack('<h', fragment[i:i+2])[0] * lfactor))), max(-32768, min(32767, int(struct.unpack('<h', fragment[i:i+2])[0] * rfactor)))) for i in range(0, len(fragment), 2))
    elif width == 4:
        return b''.join(struct.pack('<ii', max(-2147483648, min(2147483647, int(struct.unpack('<i', fragment[i:i+4])[0] * lfactor))), max(-2147483648, min(2147483647, int(struct.unpack('<i', fragment[i:i+4])[0] * rfactor)))) for i in range(0, len(fragment), 4))
    else:
        return fragment

def ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    """Rate conversion - simplified version"""
    # Simple rate conversion - just return the fragment as-is for now
    # This is a minimal implementation for compatibility
    return fragment, state

def lin2lin(fragment, width, newwidth):
    """Convert between different widths"""
    if width == newwidth:
        return fragment
    # Simplified conversion - just return as-is for compatibility
    return fragment

def lin2ulaw(fragment, width):
    """Convert linear to ulaw - simplified"""
    return fragment

def ulaw2lin(fragment, width):
    """Convert ulaw to linear - simplified"""
    return fragment

def lin2alaw(fragment, width):
    """Convert linear to alaw - simplified"""
    return fragment

def alaw2lin(fragment, width):
    """Convert alaw to linear - simplified"""
    return fragment

def rms(fragment, width):
    """Calculate RMS - simplified"""
    if width == 1:
        return sum(struct.unpack('B', fragment[i:i+1])[0] for i in range(0, len(fragment), 1)) / (len(fragment) // width)
    elif width == 2:
        return sum(abs(struct.unpack('<h', fragment[i:i+2])[0]) for i in range(0, len(fragment), 2)) / (len(fragment) // width)
    elif width == 4:
        return sum(abs(struct.unpack('<i', fragment[i:i+4])[0]) for i in range(0, len(fragment), 4)) / (len(fragment) // width)
    else:
        return 0

def cross(fragment, width):
    """Calculate cross - simplified"""
    return 0

def findfactor(fragment, reference):
    """Find factor - simplified"""
    return 1.0

def findfit(fragment, reference):
    """Find fit - simplified"""
    return 1.0, 0

def findmax(fragment, width):
    """Find maximum - simplified"""
    if width == 1:
        return max(struct.unpack('B', fragment[i:i+1])[0] for i in range(0, len(fragment), 1))
    elif width == 2:
        return max(struct.unpack('<h', fragment[i:i+2])[0] for i in range(0, len(fragment), 2))
    elif width == 4:
        return max(struct.unpack('<i', fragment[i:i+4])[0] for i in range(0, len(fragment), 4))
    else:
        return 0

def getsample(fragment, width, index):
    """Get sample - simplified"""
    if width == 1:
        return struct.unpack('B', fragment[index:index+1])[0]
    elif width == 2:
        return struct.unpack('<h', fragment[index:index+2])[0]
    elif width == 4:
        return struct.unpack('<i', fragment[index:index+4])[0]
    else:
        return 0

def avg(fragment, width):
    """Calculate average - simplified"""
    if width == 1:
        return sum(struct.unpack('B', fragment[i:i+1])[0] for i in range(0, len(fragment), 1)) / (len(fragment) // width)
    elif width == 2:
        return sum(struct.unpack('<h', fragment[i:i+2])[0] for i in range(0, len(fragment), 2)) / (len(fragment) // width)
    elif width == 4:
        return sum(struct.unpack('<i', fragment[i:i+4])[0] for i in range(0, len(fragment), 4)) / (len(fragment) // width)
    else:
        return 0

def avgpp(fragment, width):
    """Calculate average peak-to-peak - simplified"""
    return avg(fragment, width)

def max(fragment, width):
    """Find maximum - simplified"""
    return findmax(fragment, width)

def maxpp(fragment, width):
    """Find maximum peak-to-peak - simplified"""
    return findmax(fragment, width)

def minmax(fragment, width):
    """Find minimum and maximum - simplified"""
    if width == 1:
        values = [struct.unpack('B', fragment[i:i+1])[0] for i in range(0, len(fragment), 1)]
        return min(values), max(values)
    elif width == 2:
        values = [struct.unpack('<h', fragment[i:i+2])[0] for i in range(0, len(fragment), 2)]
        return min(values), max(values)
    elif width == 4:
        values = [struct.unpack('<i', fragment[i:i+4])[0] for i in range(0, len(fragment), 4)]
        return min(values), max(values)
    else:
        return 0, 0
