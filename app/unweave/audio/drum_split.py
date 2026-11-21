"""On-demand six-piece DrumSep pass. Originals survive; queue publishes atomically."""
import contextlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import tempfile

from .model_cache import DRUM_MODEL, ARTIFACTS, create_separator
from .output_files import validate_outputs
from .pipeline import require_runtime
from ..tools import cpu_threads

DRUM_STEMS = ('kick', 'snare', 'toms', 'hh', 'ride', 'crash')


def split_drums(source: Path, output: Path, cache: Path) -> list[Path]:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.is_file():
        raise ValueError('Separate the drums track first.')
    paths = [output / f'{name}.flac' for name in DRUM_STEMS]
    settings_path = output / 'separation.json'
    validate_outputs(source, paths + [settings_path])
    if any(p.exists() or p.with_suffix('.wav').exists() for p in paths):
        raise ValueError('This project already contains drum pieces.')
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError('Separation settings must contain a JSON object.')
    require_runtime(False)
    version = importlib.metadata.version('audio-separator')
    if tuple(int(x) for x in version.split('.')[:2]) != (0, 48):
        raise RuntimeError('DrumSep needs audio-separator 0.48.x. Open Settings → Components → Repair, then retry.')
    import numpy as np
    import soundfile as sf
    import torch
    info = sf.info(source)
    if info.frames <= 0 or info.samplerate != 44100 or info.channels != 2:
        raise ValueError('Drum splitting requires a nonempty stereo 44.1 kHz drums stem.')
    torch.set_num_threads(cpu_threads())
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.drum-split-', dir=output) as temporary:
        stage = Path(temporary)
        with contextlib.redirect_stdout(sys.stderr):
            print('Loading model DrumSep', file=sys.stderr, flush=True)
            separator = create_separator(output_dir=str(stage), output_format='WAV', use_soundfile=True,
                model_file_dir=str(cache), log_level=logging.WARNING,
                normalization_threshold=1.0, amplification_threshold=0.0,
                mdxc_params={'batch_size': 1, 'overlap': 4, 'process_all_stems': True})
            separator.load_model(model_filename=DRUM_MODEL)
            # Match the app's CPU policy; no dependency on a discrete GPU.
            print('Separating with DrumSep: kick, snare, toms, hh, ride, crash', file=sys.stderr, flush=True)
            separator.separate(str(source), custom_output_names={s: s for s in DRUM_STEMS})
            del separator
        print('Writing stems', file=sys.stderr, flush=True)
        for name in DRUM_STEMS:
            data, sr = sf.read(stage / f'{name}.wav', dtype='float32', always_2d=True)
            if sr != info.samplerate or data.shape != (info.frames, 2) or not np.isfinite(data).all():
                raise RuntimeError(f'DrumSep produced invalid audio for {name}.')
            # Never clip silently; keep the previous project intact on failure.
            if np.max(np.abs(data)) > 1.0:
                raise RuntimeError(f'DrumSep {name} exceeds full scale; output was not published.')
            sf.write(stage / f'{name}.flac', data, sr, subtype='PCM_24')
        settings['drum_split'] = {'model': DRUM_MODEL, 'sha256': ARTIFACTS[DRUM_MODEL][1],
                                  'stems': list(DRUM_STEMS), 'source': source.name}
        (stage/'separation.json').write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
        validate_outputs(source, paths + [settings_path])
        for path in paths + [settings_path]:
            (stage/path.name).replace(path)
    return paths
