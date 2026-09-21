"""Git-only integration checks for dependency patch preservation (no robot/GPU)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/manage_upstream.py"
REPO = SCRIPT.parents[1]

class UpstreamManagementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="upstream-test-")
        self.base = Path(self.temp.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME="Patch Test",
                        GIT_AUTHOR_EMAIL="patch-test@example.invalid",
                        GIT_COMMITTER_NAME="Patch Test",
                        GIT_COMMITTER_EMAIL="patch-test@example.invalid")
        self.origin = self.base / "origin"
        self.init(self.origin)
        (self.origin / "module.py").write_text("VALUE = 1\n")
        (self.origin / "obsolete.py").write_text("OLD = True\n")
        self.git(self.origin, "add", ".")
        self.git(self.origin, "commit", "-qm", "baseline")
        self.commit = self.git(self.origin, "rev-parse", "HEAD").strip()
        self.root = self.base / "source"
        self.init(self.root)
        (self.root / "upstream").mkdir()
        self.git(self.root, "clone", "-q", str(self.origin), "upstream/dep")
        dep = self.root / "upstream/dep"
        (dep / "module.py").write_text("VALUE = 2\n")
        (dep / "obsolete.py").unlink()
        (dep / "__init__.py").write_text("")
        (self.root / "vendor/patches").mkdir(parents=True)
        self.lock = {"version": 1, "dependencies": [
            {"name": "dep", "path": "upstream/dep", "url": str(self.origin),
             "commit": self.commit, "patch": "vendor/patches/dep.patch", "sha256": ""}
        ]}
        (self.root / "vendor/upstream.lock.json").write_text(json.dumps(self.lock))
        self.manage("export", self.root)

    def tearDown(self):
        self.temp.cleanup()

    def init(self, path):
        path.mkdir(parents=True)
        self.git(path, "init", "-q")

    def git(self, path, *args):
        return subprocess.check_output(["git", *args], cwd=path, env=self.env,
                                       text=True, stderr=subprocess.PIPE)

    def manage(self, command, root, success=True):
        result = subprocess.run([os.sys.executable, str(SCRIPT), command,
                                 "--root", str(root)], env=self.env,
                                text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def fresh(self):
        target = self.base / "fresh"
        self.init(target)
        shutil.copytree(self.root / "vendor", target / "vendor")
        return target

    def test_restores_edit_deletion_empty_file_and_is_idempotent(self):
        target = self.fresh()
        self.manage("restore", target)
        dep = target / "upstream/dep"
        self.assertEqual((dep / "module.py").read_text(), "VALUE = 2\n")
        self.assertFalse((dep / "obsolete.py").exists())
        self.assertEqual((dep / "__init__.py").read_bytes(), b"")
        before = self.git(dep, "status", "--porcelain")
        self.manage("restore", target)
        self.manage("check", target)
        self.assertEqual(before, self.git(dep, "status", "--porcelain"))

    def test_export_does_not_touch_live_index(self):
        dep = self.root / "upstream/dep"
        self.assertEqual(self.git(dep, "diff", "--cached"), "")
        self.manage("check", self.root)

    def test_linked_worktree_roundtrip_preserves_main_checkout_and_indexes(self):
        self.git(self.root, "add", "vendor")
        self.git(self.root, "commit", "-qm", "pin dependencies")
        target = self.base / "linked"
        self.git(self.root, "worktree", "add", "--detach", str(target), "HEAD")
        self.assertTrue((target / ".git").is_file())
        protected = [self.root / ".git/index", self.root / "vendor/upstream.lock.json",
                     self.root / "vendor/patches/dep.patch",
                     self.root / "upstream/dep/.git/index",
                     self.root / "upstream/dep/module.py"]
        before = {path: path.read_bytes() for path in protected}
        self.manage("restore", target)
        self.manage("check", target)
        dep = target / "upstream/dep"
        self.assertFalse((dep / "obsolete.py").exists())
        self.assertEqual((dep / "__init__.py").read_bytes(), b"")
        self.assertEqual((dep / "module.py").read_text(), "VALUE = 2\n")
        (dep / "module.py").write_text("VALUE = 3\n")
        self.git(dep, "add", "-u")
        staged = (dep / ".git/index").read_bytes()
        self.manage("export", target)
        self.manage("restore", target)
        self.manage("check", target)
        self.assertEqual((dep / ".git/index").read_bytes(), staged)
        self.assertEqual({path: path.read_bytes() for path in protected}, before)
        fresh = self.base / "linked-export-restored"
        self.init(fresh)
        shutil.copytree(target / "vendor", fresh / "vendor")
        self.manage("restore", fresh)
        self.assertEqual((fresh / "upstream/dep/module.py").read_text(), "VALUE = 3\n")

    def test_staged_deletion_is_exported_without_altering_real_index(self):
        dep = self.root / "upstream/dep"
        before = (self.root / "vendor/patches/dep.patch").read_bytes()
        self.git(dep, "add", "-u")
        staged = self.git(dep, "diff", "--cached")
        self.manage("export", self.root)
        self.assertEqual(before, (self.root / "vendor/patches/dep.patch").read_bytes())
        self.assertEqual(staged, self.git(dep, "diff", "--cached"))
        target = self.fresh()
        self.manage("restore", target)
        self.assertFalse((target / "upstream/dep/obsolete.py").exists())

    def test_staged_rename_preserves_old_path_deletion(self):
        dep = self.root / "upstream/dep"
        self.git(dep, "mv", "module.py", "renamed.py")
        self.manage("export", self.root)
        target = self.fresh()
        self.manage("restore", target)
        self.assertFalse((target / "upstream/dep/module.py").exists())
        self.assertEqual((target / "upstream/dep/renamed.py").read_text(), "VALUE = 2\n")

    def test_staged_addition_removed_from_disk_is_not_exported(self):
        dep = self.root / "upstream/dep"
        before = (self.root / "vendor/patches/dep.patch").read_bytes()
        added = dep / "removed_addition.py"
        added.write_text("TEMP = True\n")
        self.git(dep, "add", "removed_addition.py")
        added.unlink()
        self.manage("export", self.root)
        self.assertEqual(before, (self.root / "vendor/patches/dep.patch").read_bytes())
        self.assertIn("removed_addition.py", self.git(dep, "diff", "--cached", "--name-only"))

    def test_refuses_to_overwrite_unsaved_source(self):
        target = self.fresh()
        self.manage("restore", target)
        source = target / "upstream/dep/module.py"
        source.write_text("USER_WORK = True\n")
        self.manage("restore", target, success=False)
        self.assertEqual(source.read_text(), "USER_WORK = True\n")

    def test_rejects_corrupt_patch_before_cloning(self):
        target = self.fresh()
        patch = target / "vendor/patches/dep.patch"
        patch.write_bytes(patch.read_bytes() + b"corrupt")
        result = self.manage("restore", target, success=False)
        self.assertIn("checksum", result.stderr)
        self.assertFalse((target / "upstream").exists())

    def test_wrong_base_is_rejected(self):
        dep = self.root / "upstream/dep"
        self.git(dep, "add", "-A")
        self.git(dep, "commit", "-qm", "unexpected base")
        result = self.manage("check", self.root, success=False)
        self.assertIn("pinned", result.stderr)

class RepositoryManifestTest(unittest.TestCase):
    def test_recorded_patches_and_env_pins_match(self):
        lock = json.loads((REPO / "vendor/upstream.lock.json").read_text())
        pins = dict(line.split("=", 1) for line in (REPO / "env/pins.env").read_text().splitlines()
                    if "=" in line and not line.startswith("#"))
        keys = {"hil-serl": "HIL_SERL_COMMIT", "serl": "SERL_COMMIT",
                "agentlace": "AGENTLACE_COMMIT",
                "serl_franka_controllers": "SERL_FRANKA_CONTROLLERS_COMMIT"}
        for dep in lock["dependencies"]:
            self.assertEqual(hashlib.sha256((REPO / dep["patch"]).read_bytes()).hexdigest(),
                             dep["sha256"], dep["name"])
            self.assertEqual(dep["commit"], pins[keys[dep["name"]]])

if __name__ == "__main__":
    unittest.main()
