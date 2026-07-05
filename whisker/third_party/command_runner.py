# START_DIFF: whisker/core/third_party_bridge/command_runner.py [Redirect stdout/stderr to files]
import logging
from pathlib import Path
import platform
import os
import shlex
import subprocess
from typing import List, Optional
from contextlib import ExitStack

_BASE_WHISKER_DIR = Path(__file__).parent.parent.parent.parent

class CommandExecutionError(Exception):
    def __init__(
        self,
        message: str,
        return_code: Optional[int] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.return_code = return_code
        self.stderr = stderr

    def __str__(self):
        details = (
            f"Return Code: {self.return_code}"
            if self.return_code is not None
            else "Return Code: N/A"
        )
        if self.stderr:
            details += f"\nStderr:\n{self.stderr.strip()}"
        return f"{super().__str__()}\n{details}"


class CommandRunner:
    def run_script_async(
        self,
        conda_env_name: str,
        script_path: str,
        script_args: List[str],
        stdout_log_path: Optional[str] = None, # New arg
        stderr_log_path: Optional[str] = None, # New arg
    ) -> subprocess.Popen:
        """
        Starts a Python script using 'conda run' asynchronously, optionally
        redirecting stdout and stderr to files.

        Args:
            conda_env_name: Target Conda environment name.
            script_path: Absolute path to the Python script.
            script_args: List of string arguments for the script.
            stdout_log_path: Optional path to redirect stdout (appends).
            stderr_log_path: Optional path to redirect stderr (appends).

        Returns:
            The subprocess.Popen object for the running script.

        Raises:
            CommandExecutionError: If the command fails to start.
            FileNotFoundError: If the script_path does not exist.
        """
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        # DEV_NOTE: Added '-u' for unbuffered Python output, better for logging
        command = [
            "conda", "run", "-n", conda_env_name, "--no-capture-output",
            "python", "-u", script_path,
        ] + script_args

        logging.info(
            f"Preparing async command: {' '.join(shlex.quote(str(c)) for c in command)}"
        )
        if stdout_log_path: logging.info(f"  Redirecting stdout to: {stdout_log_path}")
        if stderr_log_path: logging.info(f"  Redirecting stderr to: {stderr_log_path}")

        use_shell = platform.system() == "Windows"
        logging.debug(f"Using shell={use_shell} for Popen (OS: {platform.system()})")

        # Use ExitStack to manage file handles. It ensures they are closed if Popen fails.
        # If Popen succeeds, we transfer ownership to the process by not closing them manually.
        with ExitStack() as stack:
            
            stdout_handle = subprocess.DEVNULL
            stderr_handle = subprocess.DEVNULL

            try:
                # 1. Setup STDOUT handle
                if stdout_log_path:
                    # Ensure the directory exists
                    log_dir = os.path.dirname(stdout_log_path)
                    if log_dir: # Handle cases where path might be just a filename
                        os.makedirs(log_dir, exist_ok=True)
                    # Use line buffering (bufsize=1), text mode, append
                    stdout_handle = stack.enter_context(
                        open(stdout_log_path, 'a', encoding='utf-8', buffering=1)
                    )
                
                # 2. Setup STDERR handle (redirect to STDOUT handle if stderr_log_path is None)
                if stderr_log_path:
                    # Ensure the directory exists
                    log_dir = os.path.dirname(stderr_log_path)
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)
                    stderr_handle = stack.enter_context(
                        open(stderr_log_path, 'a', encoding='utf-8', buffering=1)
                    )
                else:
                    # If stdout is redirected to a file, redirect stderr to the same file.
                    # Otherwise, it remains DEVNULL.
                    if stdout_handle is not subprocess.DEVNULL:
                        stderr_handle = stdout_handle
                    # DEV_NOTE: No need to use stack.enter_context for an existing handle (or DEVNULL).
                
                # 3. Execute Popen
                creationflags = 0
                if use_shell and platform.system() == "Windows":
                     # Prevent console window from popping up on Windows when using shell=True
                     creationflags = subprocess.CREATE_NO_WINDOW

                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=False,
                    shell=use_shell,
                    creationflags=creationflags,
                    cwd=_BASE_WHISKER_DIR,
                )

                logging.info(
                    f"Started background process PID: {process.pid} for script '{os.path.basename(script_path)}'"
                )

                # CRITICAL: ExitStack cleanup is skipped/cancelled on success, 
                # passing ownership of file handles to the Popen process.
                stack.pop_all() 

                return process

            except (FileNotFoundError, OSError) as e:
                err_msg: str
                # Check if the error is specifically about 'conda' not being found
                is_conda_not_found = isinstance(e, FileNotFoundError) and (
                    "conda" in str(e) or (isinstance(command, list) and command and "conda" == command[0])
                )
                if is_conda_not_found:
                    err_msg = "Failed to execute command: 'conda' not found. Ensure Conda is installed and in the system PATH."
                else:
                    err_msg = f"Failed to execute command: System or file error during Popen execution: {e} (Check command, path, env, shell={use_shell})"
                logging.critical(err_msg, exc_info=True)
                # ExitStack will close all opened handles automatically here.
                raise CommandExecutionError(err_msg) from e
            except Exception as e:
                # Catch other potential Popen errors (less common)
                err_msg = f"An unexpected error occurred during subprocess initialization: {type(e).__name__}: {e}"
                logging.critical(err_msg, exc_info=True)
                # ExitStack will close all opened handles automatically here.
                raise CommandExecutionError(err_msg) from e
