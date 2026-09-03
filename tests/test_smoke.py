from pathlib import Path

from edgetts_arena.adapters import DummyTTSAdapter
from edgetts_arena.utils import write_wav


def test_dummy_to_wav_smoke(tmp_path: Path) -> None:
    adapter = DummyTTSAdapter()
    adapter.load_model()
    output = adapter.infer("EdgeTTS Arena smoke test", seed=1)
    wav = write_wav(tmp_path / "dummy.wav", output.audio, output.sample_rate)
    assert wav.exists()
    assert wav.stat().st_size > 44
