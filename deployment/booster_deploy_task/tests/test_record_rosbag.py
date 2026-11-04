"""Exercise the recorder wrapper with a fake ros2; no ROS or robot connection."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = (r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt'
        else shutil.which('bash'))


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'bash not installed')
class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.capture = self.folder / 'calls.txt'
        self.output = self.folder / 'new_bag'
        mock = self.folder / 'ros2'
        mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
                        encoding='utf-8', newline='\n')
        mock.chmod(0o755)

    def run_wrapper(self, distro='humble'):
        env = dict(os.environ, ROS_DISTRO=distro, CAPTURE_FILE=self.capture.as_posix())
        return subprocess.run(
            [BASH, '--noprofile', '--norc', '-c',
             'mock_bin="$1"; if command -v cygpath >/dev/null; then mock_bin="$(cygpath -u "$mock_bin")"; fi; '
             'export PATH="$mock_bin:$PATH"; exec bash "$2" "$3"', '_',
             self.folder.as_posix(), (ROOT/'scripts/record_rosbag.sh').as_posix(),
             self.output.as_posix()],
            env=env, capture_output=True, text=True, timeout=10)

    def test_only_record_command_and_explicit_topics(self):
        result=self.run_wrapper()
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.capture.read_text().splitlines()
        self.assertEqual(args[:2],['bag','record'])
        self.assertIn('--qos-profile-overrides-path',args)
        self.assertEqual(args[args.index('--storage')+1],'sqlite3')
        self.assertEqual(args[args.index('--output')+1],self.output.as_posix())
        self.assertEqual(args[-4:],['/low_state','/joint_ctrl','/remote_controller_state','/rosout'])

    def test_existing_directory_refused_without_calling_ros(self):
        self.output.mkdir(); marker=self.output/'keep.txt'; marker.write_text('keep')
        result=self.run_wrapper()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.capture.exists())
        self.assertEqual(marker.read_text(),'keep')

    def test_wrong_distro_refused_without_calling_ros(self):
        result=self.run_wrapper('rolling')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.capture.exists())


if __name__ == '__main__':
    unittest.main()
