"""Capture all nine baseline launch commands without Slurm or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BaselineLauncherTests(unittest.TestCase):
    def test_launchers_and_overrides(self):
        repo = Path(__file__).resolve().parents[1]
        scripts = sorted((repo / 'comparison/training/scripts').glob('cineca_*_iabench.sh'))
        self.assertEqual(len(scripts), 9)
        with tempfile.TemporaryDirectory(prefix='baseline slurm ') as directory:
            work = Path(directory)
            (work / 'hyp_fine_tuning/bin').mkdir(parents=True)
            (work / 'hyp_fine_tuning/bin/activate').touch()
            checkpoint = work / 'stage one.pth'
            checkpoint.touch()
            capture = work / 'command.json'
            env = dict(os.environ, WORK=str(work), SCRATCH=str(work / 'scratch'), REPO=str(repo),
                       TEST_PYTHON=sys.executable, CAPTURE=str(capture))
            for key in ('DATA', 'SPLIT_MANIFEST', 'NUM_WORKERS', 'LOGDIR', 'RESUME_CHECKPOINT', 'STAGE_ONE_CHECKPOINT'):
                env.pop(key, None)
            stub = r'''
module() { :; }
python() {
    "$TEST_PYTHON" -c 'import json, os, sys; from pathlib import Path; Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv": sys.argv[1:], "pythonpath": os.environ["PYTHONPATH"]}))' "$@"
}
export -f module python
exec bash "$@"
'''
            for script in scripts:
                text = script.read_text()
                self.assertIn('#SBATCH --time=4-00:00:00', text)
                self.assertIn('#SBATCH --qos=boost_qos_lprod', text)
                self.assertIn('#SBATCH --gpus-per-node=1', text)
                self.assertIn('#SBATCH --cpus-per-task=8', text)
                self.assertNotIn('IAB_EXCLUDE_GENERATORS', text)
                subprocess.run(['bash', '-n', str(script)], check=True)
                stage_two = script.name == 'cineca_dna_train_iabench.sh'
                for overrides in ({}, {'DATA': str(work / 'images'), 'SPLIT_MANIFEST': str(work / 'split.json'),
                                       'NUM_WORKERS': '3', 'LOGDIR': str(work / 'logs'), 'RESUME_CHECKPOINT': str(checkpoint)}):
                    with self.subTest(script=script.name, overrides=overrides):
                        command = ['bash', '-c', stub, '_', str(script)]
                        if stage_two and not overrides:
                            command.append(str(checkpoint))
                        subprocess.run(command, env=dict(env, **overrides), check=True, capture_output=True, text=True)
                        captured = json.loads(capture.read_text())
                        argv = captured['argv']
                        self.assertEqual(argv[:2], ['-m', 'comparison.training.train'])
                        flags = dict(zip(argv[2::2], argv[3::2]))
                        self.assertEqual(flags['--dataset'], 'iabench')
                        self.assertEqual(flags['--root_dir'], overrides.get('DATA', str(work / 'scratch/datasets/IABench_images')))
                        self.assertEqual(flags['--split_manifest'], overrides.get('SPLIT_MANIFEST', str(
                            work / 'hyp_fine_tuning/checkpoints/attribution_iabench_random_vitl14.splits.json')))
                        self.assertEqual(flags['--num_workers'], overrides.get('NUM_WORKERS', '2' if 'defl' in script.name else '8'))
                        self.assertEqual(flags['--log_dir'], overrides.get('LOGDIR', str(repo / 'comparison/training/logs')))
                        self.assertEqual(flags['--batch_size'], '8' if any(m in script.name for m in ('defl', 'hifi_net')) else '32')
                        self.assertEqual(flags['--n_epoch'], '20' if 'patch' in script.name else '10')
                        if overrides:
                            self.assertEqual(flags['--resume_checkpoint'], str(checkpoint))
                            self.assertNotIn('--pretrained_path', flags)
                        elif stage_two:
                            self.assertEqual(flags['--pretrained_path'], str(checkpoint))
                        if 'dna' in script.name:
                            self.assertIn('iab_pydeps', captured['pythonpath'])
                        expected_config = ('dna_pretrain' if 'dna_pretrain' in script.name else
                                           'dna_default' if stage_two else script.name.removeprefix('cineca_').removesuffix('_train_iabench.sh'))
                        self.assertTrue(flags['--config'].endswith(f'/{expected_config}.yaml'))
            capture.unlink()
            script = repo / 'comparison/training/scripts/cineca_dna_train_iabench.sh'
            failed = subprocess.run(['bash', '-c', stub, '_', str(script)], env=env, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
