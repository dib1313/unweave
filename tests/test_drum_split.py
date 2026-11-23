"""DrumSep output safety, project timeline, and queue integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from unweave.audio.drum_split import DRUM_STEMS, split_drums
from unweave.projects import describe_project
from unweave.webui.library import Library
from unweave.cli import parser
from test_trim import project


class DrumSplitTests(unittest.TestCase):
    def test_cli(self):
        args = parser().parse_args(['split-drums', '--file', 'drums.flac', '--output', 'out'])
        self.assertEqual(args.command, 'split-drums')

    def test_listing_hides_original_only_when_all_pieces_exist_and_keeps_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            p = project(Path(temp), {'trim': {'start': 30, 'end': 40}, 'drum_split': {'model': 'test'}}, names=('drums', 'bass'))
            for name in DRUM_STEMS[:-1]:
                (p/'stems'/f'{name}.flac').write_bytes(b'audio')
            self.assertIn('drums.wav', [s['file'] for s in describe_project(p)['stems']])
            (p/'stems'/'crash.flac').write_bytes(b'audio')
            stems = describe_project(p)['stems']
            self.assertNotIn('drums.wav', [s['file'] for s in stems])
            self.assertTrue(all(s['offset'] == 30 for s in stems))
            self.assertTrue((p/'stems'/'drums.wav').exists())

    def test_success_and_failure_leave_original_untouched(self):
        for broken in (False, True):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                source = out/'drums.flac'
                sf.write(source, np.ones((4410, 2), dtype='float32')*.05, 44100)
                before = source.read_bytes()
                settings = out/'separation.json'
                settings.write_text('{"trim":{"start":30,"end":31}}')
                previous = settings.read_bytes()
                fake = mock.Mock()
                def run(*a, **kw):
                    for name in DRUM_STEMS:
                        if broken and name == 'ride':
                            continue
                        sf.write(Path(fake.output_dir)/f'{name}.wav', np.zeros((4410, 2)), 44100)
                fake.separate.side_effect = run
                def factory(**kw):
                    fake.output_dir = kw['output_dir']
                    return fake
                with mock.patch('unweave.audio.drum_split.create_separator', side_effect=factory), mock.patch('unweave.audio.drum_split.require_runtime'):
                    if broken:
                        with self.assertRaises(Exception):
                            split_drums(source, out, out/'cache')
                        self.assertEqual(previous, settings.read_bytes())
                        self.assertFalse((out/'kick.flac').exists())
                    else:
                        result = split_drums(source, out, out/'cache')
                        self.assertEqual(len(result), 6)
                        self.assertEqual(json.loads(settings.read_text())['trim']['start'], 30)
                        fake.load_model.assert_called_once()
                self.assertEqual(source.read_bytes(), before)

    def test_queue_stages_all_files_and_does_not_touch_original(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            lib = Library(base/'projects', base/'ui.json', base/'cache')
            p = project(lib.root, {}, names=('drums', 'bass'))
            with mock.patch.object(lib, '_enqueue', side_effect=lambda j:j), mock.patch('unweave.webui.queue.probe_duration', return_value=10):
                job = lib.start_drum_split('Song')
            stage = base/'stage'
            cmd = lib._separation_command(job,p,job['_original'],stage,job['_stems'],None,False,None)
            self.assertIn('split-drums', cmd)
            self.assertTrue((stage/'bass.wav').exists())
            self.assertTrue((p/'stems'/'drums.wav').exists())
            (p/'stems'/'kick.wav').write_bytes(b'RIFF')
            with self.assertRaisesRegex(ValueError, 'already contains'):
                lib.start_drum_split('Song')

    def test_missing_drums_rejected_before_enqueue(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            lib = Library(base/'projects', base/'ui.json', base/'cache')
            project(lib.root, {}, names=('vocals',))
            with self.assertRaisesRegex(ValueError, 'drums track first'):
                lib.start_drum_split('Song')
