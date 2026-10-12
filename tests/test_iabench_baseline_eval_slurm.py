"""Capture the shared baseline evaluation launcher without Slurm or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EvaluationLauncherTests(unittest.TestCase):
    def test_launch_command_defaults_overrides_and_missing_inputs(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / 'comparison/training/scripts/cineca_baselines_test_iabench.sh'
        for directive in ('--time=4-00:00:00', '--qos=boost_qos_lprod', '--gpus-per-node=1', '--cpus-per-task=8'):
            self.assertIn(f'#SBATCH {directive}', script.read_text())
        self.assertNotIn('IAB_EXCLUDE_GENERATORS', script.read_text())
        subprocess.run(['bash', '-n', str(script)], check=True)
        with tempfile.TemporaryDirectory(prefix='baseline evaluation ') as directory:
            work = Path(directory)
            base = work / 'hyp_fine_tuning'
            (base / 'bin').mkdir(parents=True)
            (base / 'bin/activate').touch()
            (base / 'checkpoints').mkdir()
            manifest = base / 'checkpoints/attribution_iabench_random_vitl14.splits.json'
            manifest.touch()
            checkpoint = work / 'selected checkpoint.pth'
            checkpoint.touch()
            capture = work / 'command.json'
            env = dict(os.environ, WORK=str(work), SCRATCH=str(work / 'scratch'), REPO=str(repo),
                       SLURM_JOB_ID='123', TEST_PYTHON=sys.executable, CAPTURE=str(capture))
            for key in ('DATA', 'SPLIT_MANIFEST', 'NUM_WORKERS', 'BATCH_SIZE', 'LOGDIR', 'CKPT', 'IAB_PYDEPS'):
                env.pop(key, None)
            stub = r'''
module() { :; }
python() {
    "$TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "pythonpath": os.environ["PYTHONPATH"]}))' "$@"
}
export -f module python
exec bash "$@"
'''
            for overrides in ({}, {'DATA': str(work / 'exported images'), 'SPLIT_MANIFEST': str(manifest),
                                    'LOGDIR': str(work / 'results'), 'NUM_WORKERS': '2', 'BATCH_SIZE': '32',
                                    'CKPT': str(checkpoint), 'IAB_PYDEPS': str(work / 'dependency overlay')}):
                command = ['bash', '-c', stub, '_', str(script)]
                if not overrides:
                    command.append(str(checkpoint))
                subprocess.run(command, env=dict(env, **overrides), check=True, capture_output=True, text=True)
                captured = json.loads(capture.read_text())
                argv = captured['argv']
                self.assertEqual(argv[:2], ['-m', 'comparison.training.test'])
                flags = dict(zip(argv[2::2], argv[3::2]))
                self.assertEqual(flags['--dataset'], 'iabench')
                self.assertEqual(flags['--resume_checkpoint'], str(checkpoint))
                self.assertEqual(flags['--root_dir'], overrides.get('DATA', str(work / 'scratch/datasets/IABench_images')))
                self.assertEqual(flags['--split_manifest'], str(manifest))
                self.assertEqual(flags['--log_dir'], overrides.get('LOGDIR', str(work / 'outputs/baselines_iabench_123')))
                self.assertEqual(flags['--num_workers'], overrides.get('NUM_WORKERS', '8'))
                self.assertEqual(flags['--batch_size'], overrides.get('BATCH_SIZE', '8'))
                self.assertEqual((flags['--level_start'], flags['--level_end']), ('0', '7'))
                self.assertIn(overrides.get('IAB_PYDEPS', 'iab_pydeps'), captured['pythonpath'])
                self.assertNotIn('--config', flags)
            for selected in ('', '/missing/checkpoint.pth', str(checkpoint)):
                capture.unlink()
                if selected == str(checkpoint):
                    manifest.unlink()
                result = subprocess.run(['bash', '-c', stub, '_', str(script)],
                                        env=dict(env, CKPT=selected), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(capture.exists())
                capture.touch()


if __name__ == '__main__':
    unittest.main()
