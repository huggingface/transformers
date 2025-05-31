# postinstall script for pywin32
#
# copies pywintypesXX.dll and pythoncomXX.dll into the system directory,
# and creates a pth file
import argparse
import glob
import os
import shutil
import sys
import sysconfig
import tempfile
import winreg

tee_f = open(
    os.path.join(
        tempfile.gettempdir(),  # Send output somewhere so it can be found if necessary...
        "pywin32_postinstall.log",
    ),
    "w",
)


class Tee:
    def __init__(self, file):
        self.f = file

    def write(self, what):
        if self.f is not None:
            try:
                self.f.write(what.replace("\n", "\r\n"))
            except OSError:
                pass
        tee_f.write(what)

    def flush(self):
        if self.f is not None:
            try:
                self.f.flush()
            except OSError:
                pass
        tee_f.flush()


sys.stderr = Tee(sys.stderr)
sys.stdout = Tee(sys.stdout)

com_modules = [
    # module_name,                      class_names
    ("win32com.servers.interp", "Interpreter"),
    ("win32com.servers.dictionary", "DictionaryPolicy"),
    ("win32com.axscript.client.pyscript", "PyScript"),
]

# Is this a 'silent' install - ie, avoid all dialogs.
# Different than 'verbose'
silent = 0

# Verbosity of output messages.
verbose = 1

root_key_name = "Software\\Python\\PythonCore\\" + sys.winver


def get_root_hkey():
    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, root_key_name, 0, winreg.KEY_CREATE_SUB_KEY
        )
        return winreg.HKEY_LOCAL_MACHINE
    except OSError:
        # Either not exist, or no permissions to create subkey means
        # must be HKCU
        return winreg.HKEY_CURRENT_USER


# Create a function with the same signature as create_shortcut
# previously provided by bdist_wininst
def create_shortcut(
    path, description, filename, arguments="", workdir="", iconpath="", iconindex=0
):
    import pythoncom
    from win32com.shell import shell

    ilink = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink,
        None,
        pythoncom.CLSCTX_INPROC_SERVER,
        shell.IID_IShellLink,
    )
    ilink.SetPath(path)
    ilink.SetDescription(description)
    if arguments:
        ilink.SetArguments(arguments)
    if workdir:
        ilink.SetWorkingDirectory(workdir)
    if iconpath or iconindex:
        ilink.SetIconLocation(iconpath, iconindex)
    # now save it.
    ipf = ilink.QueryInterface(pythoncom.IID_IPersistFile)
    ipf.Save(filename, 0)


# Support the same list of "path names" as bdist_wininst used to
def get_special_folder_path(path_name):
    from win32com.shell import shell, shellcon

    for maybe in """
        CSIDL_COMMON_STARTMENU CSIDL_STARTMENU CSIDL_COMMON_APPDATA
        CSIDL_LOCAL_APPDATA CSIDL_APPDATA CSIDL_COMMON_DESKTOPDIRECTORY
        CSIDL_DESKTOPDIRECTORY CSIDL_COMMON_STARTUP CSIDL_STARTUP
        CSIDL_COMMON_PROGRAMS CSIDL_PROGRAMS CSIDL_PROGRAM_FILES_COMMON
        CSIDL_PROGRAM_FILES CSIDL_FONTS""".split():
        if maybe == path_name:
            csidl = getattr(shellcon, maybe)
            return shell.SHGetSpecialFolderPath(0, csidl, False)
    raise ValueError(f"{path_name} is an unknown path ID")


def CopyTo(desc, src, dest):
    import win32api
    import win32con

    while 1:
        try:
            win32api.CopyFile(src, dest, 0)
            return
        except win32api.error as details:
            if details.winerror == 5:  # access denied - user not admin.
                raise
            if silent:
                # Running silent mode - just re-raise the error.
                raise
            full_desc = (
                f"Error {desc}\n\n"
                "If you have any Python applications running, "
                f"please close them now\nand select 'Retry'\n\n{details.strerror}"
            )
            rc = win32api.MessageBox(
                0, full_desc, "Installation Error", win32con.MB_ABORTRETRYIGNORE
            )
            if rc == win32con.IDABORT:
                raise
            elif rc == win32con.IDIGNORE:
                return
            # else retry - around we go again.


# We need to import win32api to determine the Windows system directory,
# so we can copy our system files there - but importing win32api will
# load the pywintypes.dll already in the system directory preventing us
# from updating them!
# So, we pull the same trick pywintypes.py does, but it loads from
# our pywintypes_system32 directory.
def LoadSystemModule(lib_dir, modname):
    # See if this is a debug build.
    import importlib.machinery
    import importlib.util

    suffix = "_d" if "_d.pyd" in importlib.machinery.EXTENSION_SUFFIXES else ""
    filename = "%s%d%d%s.dll" % (
        modname,
        sys.version_info.major,
        sys.version_info.minor,
        suffix,
    )
    filename = os.path.join(lib_dir, "pywin32_system32", filename)
    loader = importlib.machinery.ExtensionFileLoader(modname, filename)
    spec = importlib.machinery.ModuleSpec(name=modname, loader=loader, origin=filename)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)


def SetPyKeyVal(key_name, value_name, value):
    root_hkey = get_root_hkey()
    root_key = winreg.OpenKey(root_hkey, root_key_name)
    try:
        my_key = winreg.CreateKey(root_key, key_name)
        try:
            winreg.SetValueEx(my_key, value_name, 0, winreg.REG_SZ, value)
            if verbose:
                print(f"-> {root_key_name}\\{key_name}[{value_name}]={value!r}")
        finally:
            my_key.Close()
    finally:
        root_key.Close()


def UnsetPyKeyVal(key_name, value_name, delete_key=False):
    root_hkey = get_root_hkey()
    root_key = winreg.OpenKey(root_hkey, root_key_name)
    try:
        my_key = winreg.OpenKey(root_key, key_name, 0, winreg.KEY_SET_VALUE)
        try:
            winreg.DeleteValue(my_key, value_name)
            if verbose:
                print(f"-> DELETE {root_key_name}\\{key_name}[{value_name}]")
        finally:
            my_key.Close()
        if delete_key:
            winreg.DeleteKey(root_key, key_name)
            if verbose:
                print(f"-> DELETE {root_key_name}\\{key_name}")
    except OSError as why:
        winerror = getattr(why, "winerror", why.errno)
        if winerror != 2:  # file not found
            raise
    finally:
        root_key.Close()


def RegisterCOMObjects(register=True):
    import win32com.server.register

    if register:
        func = win32com.server.register.RegisterClasses
    else:
        func = win32com.server.register.UnregisterClasses
    flags = {}
    if not verbose:
        flags["quiet"] = 1
    for module, klass_name in com_modules:
        __import__(module)
        mod = sys.modules[module]
        flags["finalize_register"] = getattr(mod, "DllRegisterServer", None)
        flags["finalize_unregister"] = getattr(mod, "DllUnregisterServer", None)
        klass = getattr(mod, klass_name)
        func(klass, **flags)


def RegisterHelpFile(register=True, lib_dir=None):
    if lib_dir is None:
        lib_dir = sysconfig.get_paths()["platlib"]
    if register:
        # Register the .chm help file.
        chm_file = os.path.join(lib_dir, "PyWin32.chm")
        if os.path.isfile(chm_file):
            # This isn't recursive, so if 'Help' doesn't exist, we croak
            SetPyKeyVal("Help", None, None)
            SetPyKeyVal("Help\\Pythonwin Reference", None, chm_file)
            return chm_file
        else:
            print("NOTE: PyWin32.chm can not be located, so has not been registered")
    else:
        UnsetPyKeyVal("Help\\Pythonwin Reference", None, delete_key=True)
    return None


def RegisterPythonwin(register=True, lib_dir=None):
    """Add (or remove) Pythonwin to context menu for python scripts.
    ??? Should probably also add Edit command for pys files also.
    Also need to remove these keys on uninstall, but there's no function
    to add registry entries to uninstall log ???
    """
    import os

    if lib_dir is None:
        lib_dir = sysconfig.get_paths()["platlib"]
    classes_root = get_root_hkey()
    ## Installer executable doesn't seem to pass anything to postinstall script indicating if it's a debug build
    pythonwin_exe = os.path.join(lib_dir, "Pythonwin", "Pythonwin.exe")
    pythonwin_edit_command = pythonwin_exe + ' -edit "%1"'

    keys_vals = [
        (
            "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Pythonwin.exe",
            "",
            pythonwin_exe,
        ),
        (
            "Software\\Classes\\Python.File\\shell\\Edit with Pythonwin",
            "command",
            pythonwin_edit_command,
        ),
        (
            "Software\\Classes\\Python.NoConFile\\shell\\Edit with Pythonwin",
            "command",
            pythonwin_edit_command,
        ),
    ]

    try:
        if register:
            for key, sub_key, val in keys_vals:
                ## Since winreg only uses the character Api functions, this can fail if Python
                ##  is installed to a path containing non-ascii characters
                hkey = winreg.CreateKey(classes_root, key)
                if sub_key:
                    hkey = winreg.CreateKey(hkey, sub_key)
                winreg.SetValueEx(hkey, None, 0, winreg.REG_SZ, val)
                hkey.Close()
        else:
            for key, sub_key, val in keys_vals:
                try:
                    if sub_key:
                        hkey = winreg.OpenKey(classes_root, key)
                        winreg.DeleteKey(hkey, sub_key)
                        hkey.Close()
                    winreg.DeleteKey(classes_root, key)
                except OSError as why:
                    winerror = getattr(why, "winerror", why.errno)
                    if winerror != 2:  # file not found
                        raise
    finally:
        # tell windows about the change
        from win32com.shell import shell, shellcon

        shell.SHChangeNotify(
            shellcon.SHCNE_ASSOCCHANGED, shellcon.SHCNF_IDLIST, None, None
        )


def get_shortcuts_folder():
    if get_root_hkey() == winreg.HKEY_LOCAL_MACHINE:
        try:
            fldr = get_special_folder_path("CSIDL_COMMON_PROGRAMS")
        except OSError:
            # No CSIDL_COMMON_PROGRAMS on this platform
            fldr = get_special_folder_path("CSIDL_PROGRAMS")
    else:
        # non-admin install - always goes in this user's start menu.
        fldr = get_special_folder_path("CSIDL_PROGRAMS")

    try:
        install_group = winreg.QueryValue(
            get_root_hkey(), root_key_name + "\\InstallPath\\InstallGroup"
        )
    except OSError:
        install_group = "Python %d.%d" % (
            sys.version_info.major,
            sys.version_info.minor,
        )
    return os.path.join(fldr, install_group)


# Get the system directory, which may be the Wow64 directory if we are a 32bit
# python on a 64bit OS.
def get_system_dir():
    import win32api  # we assume this exists.

    try:
        import pythoncom
        import win32process
        from win32com.shell import shell, shellcon

        try:
            if win32process.IsWow64Process():
                return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEMX86)
            return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEM)
        except (pythoncom.com_error, win32process.error):
            return win32api.GetSystemDirectory()
    except ImportError:
        return win32api.GetSystemDirectory()


def fixup_dbi():
    # We used to have a dbi.pyd with our .pyd files, but now have a .py file.
    # If the user didn't uninstall, they will find the .pyd which will cause
    # problems - so handle that.
    import win32api
    import win32con

    pyd_name = os.path.join(os.path.dirname(win32api.__file__), "dbi.pyd")
    pyd_d_name = os.path.join(os.path.dirname(win32api.__file__), "dbi_d.pyd")
    py_name = os.path.join(os.path.dirname(win32con.__file__), "dbi.py")
    for this_pyd in (pyd_name, pyd_d_name):
        this_dest = this_pyd + ".old"
        if os.path.isfile(this_pyd) and os.path.isfile(py_name):
            try:
                if os.path.isfile(this_dest):
                    print(
                        f"Old dbi '{this_dest}' already exists - deleting '{this_pyd}'"
                    )
                    os.remove(this_pyd)
                else:
                    os.rename(this_pyd, this_dest)
                    print(f"renamed '{this_pyd}'->'{this_pyd}.old'")
            except OSError as exc:
                print(f"FAILED to rename '{this_pyd}': {exc}")


def install(lib_dir):
    import traceback

    # The .pth file is now installed as a regular file.
    # Create the .pth file in the site-packages dir, and use only relative paths
    # We used to write a .pth directly to sys.prefix - clobber it.
    if os.path.isfile(os.path.join(sys.prefix, "pywin32.pth")):
        os.unlink(os.path.join(sys.prefix, "pywin32.pth"))
    # The .pth may be new and therefore not loaded in this session.
    # Setup the paths just in case.
    for name in "win32 win32\\lib Pythonwin".split():
        sys.path.append(os.path.join(lib_dir, name))
    # It is possible people with old versions installed with still have
    # pywintypes and pythoncom registered.  We no longer need this, and stale
    # entries hurt us.
    for name in "pythoncom pywintypes".split():
        keyname = "Software\\Python\\PythonCore\\" + sys.winver + "\\Modules\\" + name
        for root in winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER:
            try:
                winreg.DeleteKey(root, keyname + "\\Debug")
            except OSError:
                pass
            try:
                winreg.DeleteKey(root, keyname)
            except OSError:
                pass
    LoadSystemModule(lib_dir, "pywintypes")
    LoadSystemModule(lib_dir, "pythoncom")
    import win32api

    # and now we can get the system directory:
    files = glob.glob(os.path.join(lib_dir, "pywin32_system32\\*.*"))
    if not files:
        raise RuntimeError("No system files to copy!!")
    # Try the system32 directory first - if that fails due to "access denied",
    # it implies a non-admin user, and we use sys.prefix
    for dest_dir in [get_system_dir(), sys.prefix]:
        # and copy some files over there
        worked = 0
        try:
            for fname in files:
                base = os.path.basename(fname)
                dst = os.path.join(dest_dir, base)
                CopyTo("installing %s" % base, fname, dst)
                if verbose:
                    print(f"Copied {base} to {dst}")
                worked = 1
                # Nuke any other versions that may exist - having
                # duplicates causes major headaches.
                bad_dest_dirs = [
                    os.path.join(sys.prefix, "Library\\bin"),
                    os.path.join(sys.prefix, "Lib\\site-packages\\win32"),
                ]
                if dest_dir != sys.prefix:
                    bad_dest_dirs.append(sys.prefix)
                for bad_dest_dir in bad_dest_dirs:
                    bad_fname = os.path.join(bad_dest_dir, base)
                    if os.path.exists(bad_fname):
                        # let exceptions go here - delete must succeed
                        os.unlink(bad_fname)
            if worked:
                break
        except win32api.error as details:
            if details.winerror == 5:
                # access denied - user not admin - try sys.prefix dir,
                # but first check that a version doesn't already exist
                # in that place - otherwise that one will still get used!
                if os.path.exists(dst):
                    msg = (
                        "The file '%s' exists, but can not be replaced "
                        "due to insufficient permissions.  You must "
                        "reinstall this software as an Administrator" % dst
                    )
                    print(msg)
                    raise RuntimeError(msg)
                continue
            raise
    else:
        raise RuntimeError(
            "You don't have enough permissions to install the system files"
        )

    # Register our demo COM objects.
    try:
        try:
            RegisterCOMObjects()
        except win32api.error as details:
            if details.winerror != 5:  # ERROR_ACCESS_DENIED
                raise
            print("You do not have the permissions to install COM objects.")
            print("The sample COM objects were not registered.")
    except Exception:
        print("FAILED to register the Python COM objects")
        traceback.print_exc()

    # There may be no main Python key in HKCU if, eg, an admin installed
    # python itself.
    winreg.CreateKey(get_root_hkey(), root_key_name)

    chm_file = None
    try:
        chm_file = RegisterHelpFile(True, lib_dir)
    except Exception:
        print("Failed to register help file")
        traceback.print_exc()
    else:
        if verbose:
            print("Registered help file")

    # misc other fixups.
    fixup_dbi()

    # Register Pythonwin in context menu
    try:
        RegisterPythonwin(True, lib_dir)
    except Exception:
        print("Failed to register pythonwin as editor")
        traceback.print_exc()
    else:
        if verbose:
            print("Pythonwin has been registered in context menu")

    # Create the win32com\gen_py directory.
    make_dir = os.path.join(lib_dir, "win32com", "gen_py")
    if not os.path.isdir(make_dir):
        if verbose:
            print(f"Creating directory {make_dir}")
        os.mkdir(make_dir)

    try:
        # create shortcuts
        # CSIDL_COMMON_PROGRAMS only available works on NT/2000/XP, and
        # will fail there if the user has no admin rights.
        fldr = get_shortcuts_folder()
        # If the group doesn't exist, then we don't make shortcuts - its
        # possible that this isn't a "normal" install.
        if os.path.isdir(fldr):
            dst = os.path.join(fldr, "PythonWin.lnk")
            create_shortcut(
                os.path.join(lib_dir, "Pythonwin\\Pythonwin.exe"),
                "The Pythonwin IDE",
                dst,
                "",
                sys.prefix,
            )
            if verbose:
                print("Shortcut for Pythonwin created")
            # And the docs.
            if chm_file:
                dst = os.path.join(fldr, "Python for Windows Documentation.lnk")
                doc = "Documentation for the PyWin32 extensions"
                create_shortcut(chm_file, doc, dst)
                if verbose:
                    print("Shortcut to documentation created")
        else:
            if verbose:
                print(f"Can't install shortcuts - {fldr!r} is not a folder")
    except Exception as details:
        print(details)

    # importing win32com.client ensures the gen_py dir created - not strictly
    # necessary to do now, but this makes the installation "complete"
    try:
        import win32com.client  # noqa
    except ImportError:
        # Don't let this error sound fatal
        pass
    print("The pywin32 extensions were successfully installed.")


def uninstall(lib_dir):
    # First ensure our system modules are loaded from pywin32_system, so
    # we can remove the ones we copied...
    LoadSystemModule(lib_dir, "pywintypes")
    LoadSystemModule(lib_dir, "pythoncom")

    try:
        RegisterCOMObjects(False)
    except Exception as why:
        print(f"Failed to unregister COM objects: {why}")

    try:
        RegisterHelpFile(False, lib_dir)
    except Exception as why:
        print(f"Failed to unregister help file: {why}")
    else:
        if verbose:
            print("Unregistered help file")

    try:
        RegisterPythonwin(False, lib_dir)
    except Exception as why:
        print(f"Failed to unregister Pythonwin: {why}")
    else:
        if verbose:
            print("Unregistered Pythonwin")

    try:
        # remove gen_py directory.
        gen_dir = os.path.join(lib_dir, "win32com", "gen_py")
        if os.path.isdir(gen_dir):
            shutil.rmtree(gen_dir)
            if verbose:
                print(f"Removed directory {gen_dir}")

        # Remove pythonwin compiled "config" files.
        pywin_dir = os.path.join(lib_dir, "Pythonwin", "pywin")
        for fname in glob.glob(os.path.join(pywin_dir, "*.cfc")):
            os.remove(fname)

        # The dbi.pyd.old files we may have created.
        try:
            os.remove(os.path.join(lib_dir, "win32", "dbi.pyd.old"))
        except OSError:
            pass
        try:
            os.remove(os.path.join(lib_dir, "win32", "dbi_d.pyd.old"))
        except OSError:
            pass

    except Exception as why:
        print(f"Failed to remove misc files: {why}")

    try:
        fldr = get_shortcuts_folder()
        for link in ("PythonWin.lnk", "Python for Windows Documentation.lnk"):
            fqlink = os.path.join(fldr, link)
            if os.path.isfile(fqlink):
                os.remove(fqlink)
                if verbose:
                    print(f"Removed {link}")
    except Exception as why:
        print(f"Failed to remove shortcuts: {why}")
    # Now remove the system32 files.
    files = glob.glob(os.path.join(lib_dir, "pywin32_system32\\*.*"))
    # Try the system32 directory first - if that fails due to "access denied",
    # it implies a non-admin user, and we use sys.prefix
    try:
        for dest_dir in [get_system_dir(), sys.prefix]:
            # and copy some files over there
            worked = 0
            for fname in files:
                base = os.path.basename(fname)
                dst = os.path.join(dest_dir, base)
                if os.path.isfile(dst):
                    try:
                        os.remove(dst)
                        worked = 1
                        if verbose:
                            print("Removed file %s" % (dst))
                    except Exception:
                        print(f"FAILED to remove {dst}")
            if worked:
                break
    except Exception as why:
        print(f"FAILED to remove system files: {why}")


# NOTE: This used to be run from inside the bdist_wininst created binary un/installer.
# From inside the binary installer this script HAD to NOT
# call sys.exit() or raise SystemExit, otherwise the installer would also terminate!
# Out of principle, we're still not using system exits.


def verify_destination(location: str) -> str:
    location = os.path.abspath(location)
    if not os.path.isdir(location):
        raise argparse.ArgumentTypeError(
            f'Path "{location}" is not an existing directory!'
        )
    return location


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""A post-install script for the pywin32 extensions.

    * Typical usage:

    > python -m pywin32_postinstall -install

    * or (shorter but you don't have control over which python environment is used)

    > pywin32_postinstall -install

    You need to execute this script, with a '-install' parameter,
    to ensure the environment is setup correctly to install COM objects, services, etc.
    """,
    )
    parser.add_argument(
        "-install",
        default=False,
        action="store_true",
        help="Configure the Python environment correctly for pywin32.",
    )
    parser.add_argument(
        "-remove",
        default=False,
        action="store_true",
        help="Try and remove everything that was installed or copied.",
    )
    parser.add_argument(
        "-wait",
        type=int,
        help="Wait for the specified process to terminate before starting.",
    )
    parser.add_argument(
        "-silent",
        default=False,
        action="store_true",
        help='Don\'t display the "Abort/Retry/Ignore" dialog for files in use.',
    )
    parser.add_argument(
        "-quiet",
        default=False,
        action="store_true",
        help="Don't display progress messages.",
    )
    parser.add_argument(
        "-destination",
        default=sysconfig.get_paths()["platlib"],
        type=verify_destination,
        help="Location of the PyWin32 installation",
    )

    args = parser.parse_args()

    if not args.quiet:
        print(f"Parsed arguments are: {args}")

    if not args.install ^ args.remove:
        parser.error("You need to either choose to -install or -remove!")

    if args.wait is not None:
        try:
            os.waitpid(args.wait, 0)
        except OSError:
            # child already dead
            pass

    silent = args.silent
    verbose = not args.quiet

    if args.install:
        install(args.destination)

    if args.remove:
        uninstall(args.destination)


if __name__ == "__main__":
    main()
