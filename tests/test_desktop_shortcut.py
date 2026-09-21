"""Tests for the desktop-shortcut installer.

Run with:  python -m unittest discover -s tests -v

The macOS bundle tests are pure file generation, so they run on any OS. The Windows
test creates real .lnk files (in a temp folder, never the real Desktop) and only runs
on Windows.
"""
import os
import plistlib
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from whisker.core.utils import desktop_shortcut as ds


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Keep the icon copy out of the real per-user data folder.
        patcher = mock.patch.object(ds, "user_data_dir", return_value=self.tmp / "userdata")
        patcher.start()
        self.addCleanup(patcher.stop)


class IconTests(TempDirCase):
    def test_ico_has_standard_sizes_up_to_256(self):
        path = ds.write_ico(self.tmp / "out" / "w.ico")
        with Image.open(path) as im:
            sizes = set(im.ico.sizes())
        self.assertIn((16, 16), sizes)
        self.assertIn((48, 48), sizes)
        self.assertIn((256, 256), sizes)

    def test_icns_is_valid_and_includes_retina_sizes(self):
        path = ds.write_icns(self.tmp / "out" / "w.icns")
        with Image.open(path) as im:
            sizes = im.info["sizes"]
        self.assertIn((512, 512, 2), sizes)
        self.assertIn((128, 128, 1), sizes)


class MacBundleTests(TempDirCase):
    python = Path("/opt/some env/bin/python")

    def test_bundle_layout_and_plist(self):
        bundle = ds.build_app_bundle(self.tmp / "WHISKER Labeler.app", self.python, self.tmp / "ws")
        with open(bundle / "Contents" / "Info.plist", "rb") as f:
            info = plistlib.load(f)
        self.assertEqual(info["CFBundleIdentifier"], ds.BUNDLE_IDENTIFIER)
        self.assertEqual(info["CFBundleExecutable"], "whisker-labeler")
        self.assertEqual(info["CFBundlePackageType"], "APPL")
        self.assertTrue((bundle / "Contents" / "Resources" / f"{info['CFBundleIconFile']}.icns").is_file())
        self.assertTrue((bundle / "Contents" / "MacOS" / "whisker-labeler").is_file())

    def test_launcher_quotes_paths_with_spaces_and_runs_the_gui(self):
        workdir = self.tmp / "WHISKER Workspace"  # a space, like a real "Application Support" path
        bundle = ds.build_app_bundle(self.tmp / "a.app", self.python, workdir)
        script = (bundle / "Contents" / "MacOS" / "whisker-labeler").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/bin/bash\n"))
        self.assertIn(f"cd {shlex.quote(str(workdir))}", script)
        self.assertIn(f"exec {shlex.quote(str(self.python))} -m whisker.main", script)
        self.assertIn("'", shlex.quote(str(workdir)))  # i.e. the space really did need quoting

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits don't exist on Windows")
    def test_launcher_is_executable(self):
        bundle = ds.build_app_bundle(self.tmp / "a.app", self.python, self.tmp)
        self.assertTrue(os.access(bundle / "Contents" / "MacOS" / "whisker-labeler", os.X_OK))

    def test_rebuilding_our_own_bundle_replaces_it(self):
        target = self.tmp / "a.app"
        ds.build_app_bundle(target, self.python, self.tmp)
        stale = target / "Contents" / "stale.txt"
        stale.write_text("old")
        ds.build_app_bundle(target, self.python, self.tmp)
        self.assertFalse(stale.exists())

    def test_refuses_to_overwrite_something_that_is_not_ours(self):
        target = self.tmp / "a.app"
        target.mkdir()
        (target / "precious.txt").write_text("keep me")
        with self.assertRaises(ds.ShortcutError):
            ds.build_app_bundle(target, self.python, self.tmp)
        self.assertTrue((target / "precious.txt").exists())

    def test_applications_only(self):
        r = ds._install_macos(False, True, self.python, self.tmp / "ws",
                              self.tmp / "Desktop", self.tmp / "Applications")
        self.assertEqual(r.created, [self.tmp / "Applications" / "WHISKER Labeler.app"])
        self.assertFalse((self.tmp / "Desktop").exists())

    def test_desktop_only_puts_the_app_on_the_desktop(self):
        r = ds._install_macos(True, False, self.python, self.tmp / "ws",
                              self.tmp / "Desktop", self.tmp / "Applications")
        self.assertEqual(r.created, [self.tmp / "Desktop" / "WHISKER Labeler.app"])
        self.assertFalse((self.tmp / "Applications").exists())

    def test_both_links_desktop_to_the_applications_copy(self):
        try:
            r = ds._install_macos(True, True, self.python, self.tmp / "ws",
                                  self.tmp / "Desktop", self.tmp / "Applications")
        except OSError as e:  # Windows without symlink privilege
            self.skipTest(f"symlinks unavailable: {e}")
        app, link = r.created
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), app.resolve())
        # Re-running replaces the old link instead of failing.
        ds._install_macos(True, True, self.python, self.tmp / "ws",
                          self.tmp / "Desktop", self.tmp / "Applications")


class ValidationTests(TempDirCase):
    def test_wrong_platform_is_rejected_with_a_helpful_message(self):
        other = next(p for p in ds.PLATFORMS if p != ds.detect_platform())
        with self.assertRaises(ds.ShortcutError) as cm:
            ds.install_shortcut(platform=other, working_dir=self.tmp)
        self.assertIn(ds.PLATFORM_LABELS[other], str(cm.exception))

    def test_unknown_platform_is_rejected(self):
        with self.assertRaises(ds.ShortcutError):
            ds.install_shortcut(platform="beos", working_dir=self.tmp)

    def test_needs_at_least_one_location(self):
        with self.assertRaises(ds.ShortcutError):
            ds.install_shortcut(desktop=False, start_menu=False, working_dir=self.tmp)


@unittest.skipUnless(sys.platform == "win32", "Windows shortcuts")
class WindowsShortcutTests(TempDirCase):
    def _read_back(self, lnk: Path) -> dict:
        script = (
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);"
            "$i=(New-Object -ComObject Shell.Application).Namespace((Split-Path $env:LNK))"
            ".ParseName((Split-Path $env:LNK -Leaf));"
            "@{target=$s.TargetPath;args=$s.Arguments;cwd=$s.WorkingDirectory;icon=$s.IconLocation;"
            "aumid=[string]$i.ExtendedProperty('System.AppUserModel.ID')}|ConvertTo-Json -Compress"
        )
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                             env={**os.environ, "LNK": str(lnk)}, capture_output=True, text=True, check=True)
        import json
        return json.loads(out.stdout)

    def test_creates_working_shortcuts_with_taskbar_identity(self):
        desktop, menu, cwd = self.tmp / "Desktop", self.tmp / "Programs", self.tmp / "My Workspace"
        r = ds.install_shortcut(desktop_dir=desktop, menu_dir=menu, working_dir=cwd)

        self.assertEqual(r.warnings, [])
        self.assertEqual(sorted(p.parent.name for p in r.created), ["Desktop", "Programs"])
        self.assertTrue(cwd.is_dir())
        for lnk in r.created:
            info = self._read_back(lnk)
            self.assertTrue(info["target"].lower().endswith("pythonw.exe"), info["target"])
            self.assertIn("-m whisker.main", info["args"])
            self.assertIn(f"--app-user-model-id {ds.APP_USER_MODEL_ID}", info["args"])
            self.assertEqual(Path(info["cwd"]), cwd)
            self.assertTrue(info["icon"].endswith("whisker.ico,0"), info["icon"])
            self.assertTrue(Path(info["icon"].rsplit(",", 1)[0]).is_file())
            self.assertEqual(info["aumid"], ds.APP_USER_MODEL_ID)

    def test_reinstall_overwrites_in_place(self):
        kwargs = dict(desktop_dir=self.tmp / "D", menu_dir=self.tmp / "P", working_dir=self.tmp / "w")
        ds.install_shortcut(**kwargs)
        r = ds.install_shortcut(**kwargs)
        self.assertEqual(len(r.created), 2)

    def test_desktop_only(self):
        r = ds.install_shortcut(start_menu=False, desktop_dir=self.tmp / "D", working_dir=self.tmp / "w")
        self.assertEqual([p.parent.name for p in r.created], ["D"])


class DialogTests(TempDirCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_install_is_disabled_for_the_other_platform(self):
        from whisker.gui.dialogs.install_shortcut_dialog import InstallShortcutDialog

        dlg = InstallShortcutDialog()
        here = ds.detect_platform()
        self.assertEqual(dlg._selected_platform(), here)  # defaults to this computer
        self.assertTrue(dlg._install_btn.isEnabled())
        self.assertEqual(dlg._notice.text(), "")

        other = next(p for p in ds.PLATFORMS if p != here)
        dlg._platform.setCurrentIndex(ds.PLATFORMS.index(other))
        self.assertFalse(dlg._install_btn.isEnabled())
        self.assertIn(ds.PLATFORM_LABELS[other], dlg._notice.text())
        self.assertEqual(dlg._menu.text(), ds.LOCATION_LABELS[other][1])  # labels follow the platform

    def test_install_passes_choices_through(self):
        from whisker.gui.dialogs import install_shortcut_dialog as mod

        dlg = mod.InstallShortcutDialog()
        dlg._menu.setChecked(False)
        result = ds.ShortcutResult(ds.detect_platform(), created=[Path("x")])
        with mock.patch.object(ds, "install_shortcut", return_value=result) as m,                 mock.patch.object(mod.QMessageBox, "information") as info:
            dlg._install()
        self.assertEqual(m.call_args.kwargs, {"platform": ds.detect_platform(), "desktop": True, "start_menu": False})
        info.assert_called_once()

    def test_errors_are_shown_not_raised(self):
        from whisker.gui.dialogs import install_shortcut_dialog as mod

        dlg = mod.InstallShortcutDialog()
        with mock.patch.object(ds, "install_shortcut", side_effect=ds.ShortcutError("boom")),                 mock.patch.object(mod.QMessageBox, "warning") as warn:
            dlg._install()
        warn.assert_called_once()
        self.assertIn("boom", warn.call_args.args[2])


class CommandLineTests(unittest.TestCase):
    def test_flags_parse(self):
        from whisker.main import WhiskerMainArgumentParser

        args, _ = WhiskerMainArgumentParser().parse_known_args(
            ["--install-shortcut", "--shortcut-location", "desktop"])
        self.assertTrue(args.install_shortcut)
        self.assertEqual(args.shortcut_location, "desktop")

        args, _ = WhiskerMainArgumentParser().parse_known_args([])
        self.assertFalse(args.install_shortcut)
        self.assertEqual(args.shortcut_location, "both")

    def test_location_maps_to_flags_and_reports_paths(self):
        from whisker.main import _install_shortcut

        result = ds.ShortcutResult("windows", created=[Path("C:/x/WHISKER Labeler.lnk")])
        for location, expected in {"both": (True, True), "desktop": (True, False), "menu": (False, True)}.items():
            with mock.patch.object(ds, "install_shortcut", return_value=result) as m, \
                    mock.patch("builtins.print") as printed:
                self.assertEqual(_install_shortcut(location), 0)
            self.assertEqual((m.call_args.kwargs["desktop"], m.call_args.kwargs["start_menu"]), expected)
            self.assertIn("WHISKER Labeler.lnk", str(printed.call_args_list[0]))

    def test_failure_returns_nonzero(self):
        from whisker.main import _install_shortcut

        with mock.patch.object(ds, "install_shortcut", side_effect=ds.ShortcutError("nope")), \
                mock.patch("builtins.print"):
            self.assertEqual(_install_shortcut("both"), 1)


if __name__ == "__main__":
    unittest.main()
