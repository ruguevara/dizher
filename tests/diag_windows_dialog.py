"""Temporary CI diagnostic: does the Windows save dialog open for a default path past MAX_PATH?"""
import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PFD = 'import sys; from imgui_bundle import portable_file_dialogs as pfd; print(repr(pfd.save_file("t", sys.argv[1]).result()))'
RAW = r'''
import ctypes, sys
from ctypes import wintypes
class OFN(ctypes.Structure):
    _fields_ = [('lStructSize', wintypes.DWORD), ('hwndOwner', wintypes.HWND), ('hInstance', wintypes.HINSTANCE),
                ('lpstrFilter', wintypes.LPCWSTR), ('lpstrCustomFilter', wintypes.LPWSTR), ('nMaxCustFilter', wintypes.DWORD),
                ('nFilterIndex', wintypes.DWORD), ('lpstrFile', wintypes.LPWSTR), ('nMaxFile', wintypes.DWORD),
                ('lpstrFileTitle', wintypes.LPWSTR), ('nMaxFileTitle', wintypes.DWORD), ('lpstrInitialDir', wintypes.LPCWSTR),
                ('lpstrTitle', wintypes.LPCWSTR), ('Flags', wintypes.DWORD), ('nFileOffset', wintypes.WORD),
                ('nFileExtension', wintypes.WORD), ('lpstrDefExt', wintypes.LPCWSTR), ('lCustData', wintypes.LPARAM),
                ('lpfnHook', ctypes.c_void_p), ('lpTemplateName', wintypes.LPCWSTR), ('pvReserved', ctypes.c_void_p),
                ('dwReserved', wintypes.DWORD), ('FlagsEx', wintypes.DWORD)]
buf = ctypes.create_unicode_buffer(sys.argv[1], 260 * 256)
ofn = OFN(lStructSize=ctypes.sizeof(OFN), lpstrFile=ctypes.cast(buf, wintypes.LPWSTR), nMaxFile=len(buf), Flags=0x80000 | 0x8)
ok = ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn))
print('GetSaveFileNameW', ok, 'CommDlgExtendedError', hex(ctypes.windll.comdlg32.CommDlgExtendedError()))
'''


def run(label, code, path):
    t = time.time()
    try:
        r = subprocess.run([sys.executable, '-c', code, path], capture_output=True, text=True, timeout=20)
        print(f'{label} {len(path)} chars: returned {r.stdout.strip()} {r.stderr.strip()[-300:]} in {time.time() - t:.1f}s')
    except subprocess.TimeoutExpired:
        print(f'{label} {len(path)} chars: dialog open (timed out)')


base = Path(tempfile.mkdtemp())
stem = 'x' * 150
build = base / stem / 'build'
build.mkdir(parents=True)
print('LongPathsEnabled (ntdll):', ctypes.windll.ntdll.RtlAreLongPathsEnabled())
print('build folder', len(str(build)), 'chars')
try:
    open(build / (stem + '.scr'), 'wb').close()
    print('plain open past MAX_PATH: ok')
except OSError as e:
    print('plain open past MAX_PATH:', e)
for name in ('short.scr', stem + '.scr'):
    run('pfd', PFD, str(build / name))
    run('raw', RAW, str(build / name))
