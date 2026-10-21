"""Task selection must follow the JSON, including paired originals and language."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

import h5py

BATCH = Path(__file__).resolve().parents[1] / 'collect_matched_libero_goal_multi_pick_place_targets.sbatch'
source = BATCH.read_text().split("<<'MATCHED_COLLECTOR_PY'\n", 1)[1].rsplit('\nMATCHED_COLLECTOR_PY', 1)[0]
job = ModuleType('multi_pick_place_collector')
with patch.dict(os.environ, MATCHED_PROJECT_ROOT=str(BATCH.parents[2])):
    exec(compile(source, str(BATCH), 'exec'), job.__dict__)


class TargetListTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = json.loads(job.DEFAULT_TARGETS.read_text())

    def load(self, data):
        path = self.root / 'targets.json'
        path.write_text(json.dumps(data))
        return job.load_targets(path)

    def test_exact_list_including_original_bowls_and_paired_paths(self):
        targets = self.load(self.raw)
        self.assertEqual(len(targets), 11)
        self.assertEqual({t['task_id'] for t in targets.values()},
                         {12, 17, 54, 55, 56, 57, 77, 78, 79, 80, 81})
        for key, entry in targets.items():
            self.assertEqual(Path(entry['bddl_path']),
                             (job.PROJECT_ROOT / self.raw[key]['bddl_path']).resolve())
            self.assertIn('paired_original_extended', entry['bddl_path'])
            self.assertEqual(entry['variants'], job.lowercase_language(self.raw[key]['variants']))

    def test_all_instruction_text_is_lowercase_including_originals(self):
        for key, entry in self.raw.items():
            if not key.startswith('_'):
                entry['base_command'] = entry['base_command'].capitalize()
                for variants in entry['variants'].values():
                    for kind, instructions in variants.items():
                        variants[kind] = [text.capitalize() for text in instructions]
        targets = self.load(self.raw)
        for entry in targets.values():
            self.assertEqual(entry['base_command'], entry['base_command'].lower())
            for variants in entry['variants'].values():
                for kind, instructions in variants.items():
                    self.assertIn(kind, {'S-PROP', 'S-LANG', 'S-MO', 'S-INT', 'S-AFF'})
                    for instruction in instructions:
                        self.assertEqual(instruction, instruction.lower())

    def test_duplicate_id_rejected_before_export_can_overwrite(self):
        first = next(v for k, v in self.raw.items() if not k.startswith('_'))
        self.raw['duplicate'] = first
        with self.assertRaisesRegex(ValueError, 'duplicate task_id'):
            self.load(self.raw)

    def test_unknown_policy_and_missing_bddl_rejected(self):
        entry = next(v for k, v in self.raw.items() if not k.startswith('_'))
        with self.assertRaisesRegex(ValueError, 'unsupported task_id'):
            self.load({'task': dict(entry, task_id=999)})
        with self.assertRaisesRegex(ValueError, 'BDDL not found'):
            self.load({'task': dict(entry, bddl_path='missing.bddl')})

    def test_no_tasks_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no tasks'):
            self.load({'_comment': ['documentation']})

    def test_exports_preserve_task_keys_and_pick_place_groups(self):
        targets = self.load(self.raw)
        for entry in targets.values():
            with h5py.File(self.root / f"task{entry['task_id']}.hdf5", 'w') as file:
                file.create_group('data/demo_1')
        job.annotate_exports(self.root, targets)
        for key, entry in targets.items():
            with h5py.File(self.root / f"task{entry['task_id']}.hdf5") as file:
                for group in (file['data'], file['data/demo_1']):
                    self.assertEqual(group.attrs['task_key'], key)
                    self.assertEqual(group.attrs['base_command'], entry['base_command'])
                    self.assertEqual(json.loads(group.attrs['variants']), entry['variants'])


if __name__ == '__main__':
    unittest.main()
