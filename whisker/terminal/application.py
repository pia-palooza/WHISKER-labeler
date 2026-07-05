import argparse
import logging
import sys
import json
import numpy as np
from pathlib import Path
from typing import Any, Optional

from whisker.base.logger import configure_console_logger, shutdown
from whisker.core.workspace import Workspace

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Application:
    def __init__(self, argv: list[str]):
        self.argv = argv
        self._json_output = "--json" in argv
        self._capture_mode = False
        self._last_result = None
        
        # Setup logging - if --json, log to stderr only to keep stdout clean for JSON
        log_level = logging.INFO
        if self._json_output:
            configure_console_logger(level=log_level, stream=sys.stderr)
        else:
            configure_console_logger(level=log_level)

    def shutdown(self):
        shutdown()

    def _output(self, data: Any):
        if self._capture_mode:
            self._last_result = {"status": "success", "data": data}
            return

        if self._json_output:
            print(json.dumps(data, indent=2, cls=NumpyEncoder))
        else:
            # Fallback for human-readable output if not --json
            if isinstance(data, dict) and "message" in data:
                print(data["message"])
            else:
                print(data)

    def _error(self, message: str, code: int = 1):
        if self._capture_mode:
            self._last_result = {"status": "error", "message": message, "code": code}
            raise SystemExit(code)

        if self._json_output:
            print(json.dumps({"status": "error", "message": message, "code": code}, indent=2, cls=NumpyEncoder))
        else:
            logging.error(message)
        sys.exit(code)

    def _create_parser(self):
        parser = argparse.ArgumentParser(prog="whisker", description="WHISKER Headless CLI")
        parser.add_argument("--json", action="store_true", help="Emit output as JSON to stdout")
        
        subparsers = parser.add_subparsers(dest="command", help="Command groups")

        # Workspace
        ws_parser = subparsers.add_parser("workspace", help="Workspace management")
        ws_sub = ws_parser.add_subparsers(dest="subcommand")
        
        open_parser = ws_sub.add_parser("open", help="Open a workspace")
        open_parser.add_argument("path", type=str)
        
        create_parser = ws_sub.add_parser("create", help="Create a workspace")
        create_parser.add_argument("path", type=str)
        create_parser.add_argument("--force", action="store_true")
        
        delete_parser = ws_sub.add_parser("delete", help="Delete a workspace")
        delete_parser.add_argument("path", type=str)
        delete_parser.add_argument("--force", action="store_true")
        
        ws_sub.add_parser("info", help="Workspace info")

        # Project
        proj_parser = subparsers.add_parser("project", help="Project management")
        proj_sub = proj_parser.add_subparsers(dest="subcommand")
        proj_sub.add_parser("list", help="List projects")
        
        proj_create = proj_sub.add_parser("create", help="Create project")
        proj_create.add_argument("name", type=str)
        proj_create.add_argument("--body_parts", type=str)
        proj_create.add_argument("--identities", type=str)
        proj_create.add_argument("--skeleton", type=str)
        proj_create.add_argument("--behaviors", type=str)
        
        proj_inspect = proj_sub.add_parser("inspect", help="Inspect project")
        proj_inspect.add_argument("name", type=str)
        
        proj_delete = proj_sub.add_parser("delete", help="Delete project")
        proj_delete.add_argument("name", type=str)

        # Dataset
        ds_parser = subparsers.add_parser("dataset", help="Dataset management")
        ds_sub = ds_parser.add_subparsers(dest="subcommand")
        ds_sub.add_parser("list", help="List datasets")
        
        ds_import = ds_sub.add_parser("import", help="Import dataset")
        ds_import.add_argument("name", type=str)
        ds_import.add_argument("type", type=str)
        ds_import.add_argument("path", type=str)
        
        ds_inspect = ds_sub.add_parser("inspect", help="Inspect dataset")
        ds_inspect.add_argument("name", type=str)
        
        ds_delete = ds_sub.add_parser("delete", help="Delete dataset")
        ds_delete.add_argument("name", type=str)

        # Run
        run_parser = subparsers.add_parser("run", help="Job execution")
        run_sub = run_parser.add_subparsers(dest="subcommand")
        run_sub.add_parser("list", help="List available jobs")
        
        run_schema = run_sub.add_parser("schema", help="Get job schema")
        run_schema.add_argument("job_name", type=str)
        
        run_exec = run_sub.add_parser("exec", help="Execute a job")
        run_exec.add_argument("job_name", type=str)
        run_exec.add_argument("--params", type=str, help="JSON string of parameters")
        run_exec.add_argument("--params_path", type=str, help="Path to a JSON file containing parameters")
        
        # System
        subparsers.add_parser("version", help="Version info")
        
        status_parser = subparsers.add_parser("status", help="Job status")
        status_parser.add_argument("job_id", nargs="?", type=str)
        
        inspect_parser = subparsers.add_parser("inspect", help="Generic inspect")
        inspect_parser.add_argument("id", type=str)

        # Batch
        batch_parser = subparsers.add_parser("batch", help="Batch command execution")
        batch_parser.add_argument("file", type=str, help="File containing commands (one per line)")

        # Pose Eval
        eval_parser = subparsers.add_parser("pose-eval", help="Pose estimation evaluation results")
        eval_sub = eval_parser.add_subparsers(dest="subcommand")
        eval_sub.add_parser("list", help="List all evaluation results")
        
        eval_get = eval_sub.add_parser("get", help="Get specific evaluation results")
        eval_get.add_argument("run_name", type=str)
        eval_get.add_argument("dataset_name", type=str)

        # Pose Model
        model_parser = subparsers.add_parser("pose-model", help="Pose estimation models")
        model_sub = model_parser.add_subparsers(dest="subcommand")
        model_sub.add_parser("list", help="List all pose models")
        
        model_inspect = model_sub.add_parser("inspect", help="Inspect model training info")
        model_inspect.add_argument("model_name", type=str)

        # Pose Training
        train_parser = subparsers.add_parser("pose-training", help="Pose estimation training")
        train_sub = train_parser.add_subparsers(dest="subcommand")
        
        run_train = train_sub.add_parser("run", help="Start a training run")
        run_train.add_argument("--run_name", required=True, type=str)
        run_train.add_argument("--datasets", required=True, type=str, help="Comma-separated dataset names")
        run_train.add_argument("--individuals", required=True, type=str, help="Comma-separated identity names")
        run_train.add_argument("--bodyparts", required=True, type=str, help="Comma-separated body part names")
        run_train.add_argument("--backend", type=str, default="WHISKER", choices=["WHISKER", "DLC"])
        run_train.add_argument("--epochs", type=int, default=100)
        run_train.add_argument("--batch_size", type=int, default=16)
        run_train.add_argument("--resume", type=str, help="Run name to resume from")
        
        # ROI Template management
        roi_template_parser = subparsers.add_parser("roi-template", help="ROI template management")
        roi_template_sub = roi_template_parser.add_subparsers(dest="subcommand")
        
        template_get = roi_template_sub.add_parser("get", help="Get ROI template for a dataset")
        template_get.add_argument("dataset_name", type=str)
        
        template_set = roi_template_sub.add_parser("set", help="Set ROI template for a dataset")
        template_set.add_argument("dataset_name", type=str)
        template_set.add_argument("--template_path", type=str, required=True)

        return parser


    def run_command(self, argv: list[str]) -> dict[str, Any]:
        """Runs a command and returns the result as a dict."""
        self._capture_mode = True
        self._last_result = None
        
        # We need to handle ArgumentParser's tendency to exit on --help or error
        parser = self._create_parser()
        
        try:
            # Capture stdout/stderr for help messages
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            f_stdout = io.StringIO()
            f_stderr = io.StringIO()
            
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                args, unknown = parser.parse_known_args(argv)
                
                if not args.command:
                    parser.print_help()
                    return {"status": "success", "message": f_stdout.getvalue(), "data": None}

                self._handle_command(args, unknown)
            
            return self._last_result or {"status": "success", "data": None}
            
        except SystemExit as e:
            if self._last_result:
                return self._last_result
            
            # This happens for --help or parsing errors
            stdout_val = f_stdout.getvalue()
            stderr_val = f_stderr.getvalue()
            
            if e.code == 0:
                return {"status": "success", "message": stdout_val, "data": None}
            else:
                return {"status": "error", "message": stderr_val or stdout_val, "code": e.code}
        except Exception as e:
            return {"status": "error", "message": str(e), "code": 1}
        finally:
            self._capture_mode = False

    def exec(self):
        parser = self._create_parser()
        args, unknown = parser.parse_known_args(self.argv)

        if not args.command:
            parser.print_help()
            return

        self._handle_command(args, unknown)

    def _handle_command(self, args: argparse.Namespace, unknown: list[str]):
        from .command_handlers import (
            WorkspaceHandler, ProjectHandler, DatasetHandler, RunHandler, 
            SystemHandler, PoseEvalHandler, PoseModelHandler, PoseTrainingHandler,
            RoiTemplateHandler
        )
        
        try:
            if args.command == "workspace":
                WorkspaceHandler(self).handle(args)
            elif args.command == "project":
                ProjectHandler(self).handle(args)
            elif args.command == "dataset":
                DatasetHandler(self).handle(args)
            elif args.command == "run":
                RunHandler(self).handle(args, unknown)
            elif args.command == "roi-template":
                RoiTemplateHandler(self).handle(args)
            elif args.command == "pose-eval":
                PoseEvalHandler(self).handle(args)
            elif args.command == "pose-model":
                PoseModelHandler(self).handle(args)
            elif args.command == "pose-training":
                PoseTrainingHandler(self).handle(args)
            elif args.command == "version":
                SystemHandler(self).version()
            elif args.command == "status":
                SystemHandler(self).status(args.job_id)
            elif args.command == "inspect":
                SystemHandler(self).inspect(args.id)
            elif args.command == "batch":
                import shlex
                with open(args.file, "r") as f:
                    commands = [shlex.split(line.strip()) for line in f if line.strip()]
                results = []
                for cmd_args in commands:
                    results.append(self.run_command(cmd_args))
                self._output(results)
            else:
                self._error(f"Unknown command: {args.command}")
        except Exception as e:
            import traceback
            logging.debug(traceback.format_exc())
            self._error(str(e))
