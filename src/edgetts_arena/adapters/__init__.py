from .dummy_adapter import DummyTTSAdapter
from .piper_adapter import PiperTTSAdapter
from .kokoro_adapter import KokoroTTSAdapter
from .qwen3_adapter import Qwen3TTSAdapter
from .qwen3_native_adapter import Qwen3NativeTTSAdapter
from .cosyvoice_adapter import CosyVoiceTTSAdapter
from .melotts_adapter import MeloTTSAdapter

__all__ = [
    "DummyTTSAdapter",
    "PiperTTSAdapter",
    "KokoroTTSAdapter",
    "Qwen3TTSAdapter",
    "Qwen3NativeTTSAdapter",
    "CosyVoiceTTSAdapter",
    "MeloTTSAdapter",
]
