# Add matcha to path before any imports
import os
import sys

# Calculate path to Matcha-TTS
_current_file = os.path.abspath(__file__)
# Go up from src_cosyvoice/__init__.py to QuadVox, then to ../CosyVoice/third_party/Matcha-TTS
_quadvox_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))))
_matcha_path = os.path.join(os.path.dirname(_quadvox_root), 'CosyVoice', 'third_party', 'Matcha-TTS')
_matcha_path = os.path.abspath(_matcha_path)

if os.path.exists(_matcha_path) and _matcha_path not in sys.path:
    sys.path.insert(0, _matcha_path)
