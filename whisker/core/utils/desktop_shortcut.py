"""Create a double-clickable launcher for the WHISKER Labeler GUI.

Windows  A ``.lnk`` shortcut on the Desktop and in the Start Menu that runs
         ``pythonw.exe -m whisker.main`` (no console window). Right-click the
         Start Menu entry and choose "Pin to taskbar" to pin it there.
macOS    A ``WHISKER Labeler.app`` bundle in ``~/Applications`` (Launchpad /
         Spotlight) plus a link to it on the Desktop. Drag it to the Dock to keep it.

Shortcuts can only be created for the OS the app is running on, because they
point at the Python interpreter that is running right now. Run the installer again
if that environment is ever moved or rebuilt.
"""
from __future__ import annotations

import logging
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

APP_NAME = "WHISKER Labeler"
APP_USER_MODEL_ID = "WHISKER.Labeler"
BUNDLE_IDENTIFIER = "org.whisker.labeler"

PLATFORMS = ("windows", "macos")
PLATFORM_LABELS = {"windows": "Windows", "macos": "macOS"}
# (desktop label, start-menu label) per platform, for UI text.
LOCATION_LABELS = {
    "windows": ("Desktop", "Start Menu"),
    "macos": ("Desktop", "Applications folder (Launchpad / Spotlight)"),
}

_ICON_SOURCE = Path(__file__).resolve().parents[2] / "gui" / "assets" / "favicon.ico"
_CREATE_NO_WINDOW = 0x08000000


class ShortcutError(RuntimeError):
    """The shortcut could not be created."""


@dataclass
class ShortcutResult:
    platform: str
    created: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def detect_platform() -> Optional[str]:
    """Return 'windows' / 'macos' for the running OS, or None if unsupported."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return None


def user_data_dir() -> Path:
    """Per-user folder for WHISKER's own files (icon copies, startup log)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "WHISKER"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "WHISKER"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "whisker"


def startup_log_path() -> Path:
    return user_data_dir() / "whisker-startup.log"


def default_working_dir() -> Path:
    """Where launched shortcuts start, so the app's default workspace (the current
    folder on first run) is a dedicated folder rather than the Desktop."""
    return Path.home() / "WHISKER Workspace"


# --------------------------------------------------------------------------- icons

def _logo_master(size: int):
    from PIL import Image

    with Image.open(_ICON_SOURCE) as src:
        largest = max(src.ico.sizes())
        frame = src.ico.getimage(largest).convert("RGBA")
    return frame.resize((size, size), Image.LANCZOS)


def write_ico(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    _logo_master(256).save(dest, format="ICO", sizes=sizes)
    return dest


def write_icns(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _logo_master(512).save(dest, format="ICNS")
    return dest


# ------------------------------------------------------------------------- public

def install_shortcut(
    platform: Optional[str] = None,
    desktop: bool = True,
    start_menu: bool = True,
    *,
    python: Optional[Path] = None,
    working_dir: Optional[Path] = None,
    desktop_dir: Optional[Path] = None,
    menu_dir: Optional[Path] = None,
) -> ShortcutResult:
    """Create the launcher. ``desktop_dir`` / ``menu_dir`` override the target
    folders (used by tests); by default the user's real ones are looked up."""
    current = detect_platform()
    platform = platform or current
    if platform not in PLATFORMS:
        raise ShortcutError(
            "Desktop shortcuts are supported on Windows and macOS only."
            if platform is None
            else f"Unknown platform '{platform}'. Choose one of: {', '.join(PLATFORMS)}."
        )
    if platform != current:
        raise ShortcutError(
            f"This computer is running {PLATFORM_LABELS.get(current, sys.platform)}, so a "
            f"{PLATFORM_LABELS[platform]} shortcut can't be created here. Run this on "
            f"the {PLATFORM_LABELS[platform]} computer instead."
        )
    if not (desktop or start_menu):
        raise ShortcutError("Choose at least one place to put the shortcut.")

    working_dir = Path(working_dir) if working_dir else default_working_dir()
    working_dir.mkdir(parents=True, exist_ok=True)
    python = Path(python) if python else Path(sys.executable)

    if platform == "windows":
        result = _install_windows(desktop, start_menu, python, working_dir, desktop_dir, menu_dir)
    else:
        result = _install_macos(desktop, start_menu, python, working_dir, desktop_dir, menu_dir)
    for path in result.created:
        logging.info("Created WHISKER launcher: %s", path)
    return result


# ------------------------------------------------------------------------ Windows

# One PowerShell run creates every requested .lnk. Paths arrive through environment
# variables so nothing needs quoting. The C# block writes the shortcut through the
# COM shell-link interface so it can also stamp the AppUserModelID (which is what
# lets a pinned taskbar icon and the running window merge into one). If Add-Type is
# blocked (e.g. constrained language mode on managed machines) we fall back to
# WScript.Shell, which cannot set the ID.
_POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'

$csharp = @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

public static class WhiskerShortcut
{
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    private class CShellLink { }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
    private interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszFile, int cch, IntPtr pfd, uint fFlags);
        void GetIDList(out IntPtr ppidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszName, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszDir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string pszDir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszArgs, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string pszArgs);
        void GetHotkey(out short pwHotkey);
        void SetHotkey(short wHotkey);
        void GetShowCmd(out int piShowCmd);
        void SetShowCmd(int iShowCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszIconPath, int cch, out int piIcon);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string pszIconPath, int iIcon);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string pszPathRel, uint dwReserved);
        void Resolve(IntPtr hwnd, uint fFlags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string pszFile);
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct PROPERTYKEY { public Guid fmtid; public uint pid; }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    private struct PROPVARIANT
    {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr pointerValue;
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    private interface IPropertyStore
    {
        uint GetCount(out uint cProps);
        uint GetAt(uint iProp, out PROPERTYKEY pkey);
        uint GetValue(ref PROPERTYKEY key, out PROPVARIANT pv);
        uint SetValue(ref PROPERTYKEY key, ref PROPVARIANT pv);
        uint Commit();
    }

    private const ushort VT_LPWSTR = 31;

    [DllImport("ole32.dll")]
    private static extern int PropVariantClear(ref PROPVARIANT pvar);

    public static void Create(string linkPath, string target, string arguments,
                              string workingDir, string icon, string description, string aumid)
    {
        IShellLinkW link = (IShellLinkW)new CShellLink();
        link.SetPath(target);
        link.SetArguments(arguments);
        link.SetWorkingDirectory(workingDir);
        link.SetIconLocation(icon, 0);
        link.SetDescription(description);
        link.SetShowCmd(1);

        if (!string.IsNullOrEmpty(aumid))
        {
            IPropertyStore store = (IPropertyStore)link;
            PROPERTYKEY key = new PROPERTYKEY();
            key.fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");
            key.pid = 5;
            PROPVARIANT value = new PROPVARIANT();
            value.vt = VT_LPWSTR;
            value.pointerValue = Marshal.StringToCoTaskMemUni(aumid);
            try
            {
                Marshal.ThrowExceptionForHR((int)store.SetValue(ref key, ref value));
                Marshal.ThrowExceptionForHR((int)store.Commit());
            }
            finally { PropVariantClear(ref value); }
        }

        ((IPersistFile)link).Save(linkPath, true);
    }
}
'@

$name    = $env:WHISKER_LNK_NAME
$target  = $env:WHISKER_LNK_TARGET
$linkArgs = $env:WHISKER_LNK_ARGS
$cwd     = $env:WHISKER_LNK_CWD
$icon    = $env:WHISKER_LNK_ICON
$desc    = $env:WHISKER_LNK_DESC
$aumid   = $env:WHISKER_LNK_AUMID

$richOk = $true
try { Add-Type -TypeDefinition $csharp } catch { $richOk = $false }

foreach ($entry in ($env:WHISKER_LNK_TARGETS -split "`n")) {
    $entry = $entry.Trim()
    if (-not $entry) { continue }
    if ($entry.StartsWith('@')) {
        $dir = [Environment]::GetFolderPath($entry.Substring(1))
        if (-not $dir) { throw "Could not locate the $($entry.Substring(1)) folder." }
    } else {
        $dir = $entry
    }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $linkPath = Join-Path $dir ($name + '.lnk')

    $status = 'basic'
    if ($richOk) {
        try {
            [WhiskerShortcut]::Create($linkPath, $target, $linkArgs, $cwd, $icon, $desc, $aumid)
            $status = 'full'
        } catch { $status = 'basic' }
    }
    if ($status -eq 'basic') {
        # No AppUserModelID available, so leave the flag that sets one on the process out.
        $shell = New-Object -ComObject WScript.Shell
        $lnk = $shell.CreateShortcut($linkPath)
        $lnk.TargetPath = $target
        $lnk.Arguments = '-m whisker.main'
        $lnk.WorkingDirectory = $cwd
        $lnk.IconLocation = $icon
        $lnk.Description = $desc
        $lnk.Save()
    }
    Write-Output ("RESULT|{0}|{1}" -f $status, $linkPath)
}
"""


def _windows_gui_python(python: Path) -> Path:
    pythonw = python.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else python


def _install_windows(
    desktop: bool,
    start_menu: bool,
    python: Path,
    working_dir: Path,
    desktop_dir: Optional[Path],
    menu_dir: Optional[Path],
) -> ShortcutResult:
    result = ShortcutResult("windows")
    icon = write_ico(user_data_dir() / "whisker.ico")

    targets: List[str] = []
    if desktop:
        targets.append(str(desktop_dir) if desktop_dir else "@Desktop")
    if start_menu:
        targets.append(str(menu_dir) if menu_dir else "@Programs")

    env = dict(os.environ)
    env.update(
        WHISKER_LNK_NAME=APP_NAME,
        WHISKER_LNK_TARGET=str(_windows_gui_python(python)),
        WHISKER_LNK_ARGS=f"-m whisker.main --app-user-model-id {APP_USER_MODEL_ID}",
        WHISKER_LNK_CWD=str(working_dir),
        WHISKER_LNK_ICON=str(icon),
        WHISKER_LNK_DESC="Launch the WHISKER Labeler",
        WHISKER_LNK_AUMID=APP_USER_MODEL_ID,
        WHISKER_LNK_TARGETS="\n".join(targets),
    )

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "make_shortcut.ps1"
        script_path.write_text(_POWERSHELL_SCRIPT, encoding="utf-8-sig")
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
                env=env, capture_output=True, text=True, timeout=120,
                creationflags=_CREATE_NO_WINDOW,
            )
        except FileNotFoundError as e:
            raise ShortcutError("PowerShell was not found, so the shortcut could not be created.") from e
        except subprocess.TimeoutExpired as e:
            raise ShortcutError("Creating the shortcut timed out.") from e

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        raise ShortcutError(f"Creating the shortcut failed:\n{detail}")

    degraded = False
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT|"):
            _, status, path = line.split("|", 2)
            result.created.append(Path(path))
            degraded = degraded or status != "full"
    if not result.created:
        raise ShortcutError("Creating the shortcut failed: PowerShell reported no shortcuts.")
    if degraded:
        result.warnings.append(
            "Windows blocked the advanced shortcut setup on this computer, so the shortcut was "
            "created in basic form. It launches WHISKER normally, but a copy pinned to the "
            "taskbar may show a second icon while the app is running."
        )
    return result


# -------------------------------------------------------------------------- macOS

def _bundle_is_ours(path: Path) -> bool:
    try:
        with open(path / "Contents" / "Info.plist", "rb") as f:
            return plistlib.load(f).get("CFBundleIdentifier") == BUNDLE_IDENTIFIER
    except (OSError, plistlib.InvalidFileException, ValueError):
        return False


def _clear_existing(path: Path) -> None:
    """Remove a previous launcher at ``path``, but never anything that isn't ours."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir() and _bundle_is_ours(path):
            shutil.rmtree(path)
        else:
            raise ShortcutError(
                f"'{path}' already exists and isn't a WHISKER launcher. Rename or remove it, "
                "then try again."
            )


def _app_version() -> str:
    try:
        from importlib.metadata import version

        return version("whisker-labeler")
    except Exception:
        return "0.1.0"


def build_app_bundle(bundle: Path, python: Path, working_dir: Path) -> Path:
    """Write ``WHISKER Labeler.app`` at ``bundle``."""
    _clear_existing(bundle)
    macos_dir = bundle / "Contents" / "MacOS"
    resources_dir = bundle / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    launcher = macos_dir / "whisker-labeler"
    launcher.write_text(
        "#!/bin/bash\n"
        f"mkdir -p {shlex.quote(str(working_dir))}\n"
        f"cd {shlex.quote(str(working_dir))} || exit 1\n"
        f'exec {shlex.quote(str(python))} -m whisker.main "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    write_icns(resources_dir / "whisker.icns")

    version = _app_version()
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
        "CFBundleExecutable": "whisker-labeler",
        "CFBundleIconFile": "whisker",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "NSHighResolutionCapable": True,
    }
    with open(bundle / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    return bundle


def _install_macos(
    desktop: bool,
    start_menu: bool,
    python: Path,
    working_dir: Path,
    desktop_dir: Optional[Path],
    apps_dir: Optional[Path],
) -> ShortcutResult:
    result = ShortcutResult("macos")
    desktop_dir = Path(desktop_dir) if desktop_dir else Path.home() / "Desktop"
    apps_dir = Path(apps_dir) if apps_dir else Path.home() / "Applications"
    bundle_name = f"{APP_NAME}.app"

    if start_menu:
        apps_dir.mkdir(parents=True, exist_ok=True)
        bundle = build_app_bundle(apps_dir / bundle_name, python, working_dir)
        result.created.append(bundle)
        if desktop:
            desktop_dir.mkdir(parents=True, exist_ok=True)
            link = desktop_dir / bundle_name
            _clear_existing(link)
            link.symlink_to(bundle, target_is_directory=True)
            result.created.append(link)
    else:
        # Desktop only: the app itself lives on the Desktop.
        desktop_dir.mkdir(parents=True, exist_ok=True)
        result.created.append(build_app_bundle(desktop_dir / bundle_name, python, working_dir))
    return result
