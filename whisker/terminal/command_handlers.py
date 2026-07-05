import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, List

from whisker.core.utils.ascii_banner import VERSION_NUMBER
from whisker.core.workspace import Workspace
from whisker.core.study.dataset import DatasetType
from whisker.core.workers.registry import JobRegistry


if TYPE_CHECKING:
    from .application import Application

from .session import SessionManager

class BaseHandler:
    def __init__(self, app: 'Application'):
        self.app = app

    def _get_workspace(self) -> Workspace:
        path = SessionManager.get_active_workspace_path()
        if not path:
            self.app._error("No active workspace. Use 'whisker workspace open <path>'", code=3)
        try:
            return Workspace(path)
        except Exception as e:
            self.app._error(f"Failed to load workspace at {path}: {e}", code=3)

class WorkspaceHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'workspace'. Use 'open', 'create', 'delete', or 'info'.")
            return

        if args.subcommand == "open":
            path = Path(args.path)
            if not path.exists():
                 self.app._error(f"Path does not exist: {args.path}", code=3)
            ws = Workspace(path)
            SessionManager.set_active_workspace_path(path)
            self.app._output({"status": "success", "message": f"Opened workspace at {args.path}", "path": str(ws.base_dir)})
        elif args.subcommand == "create":
            path = Path(args.path)
            ws = Workspace.create(path, warn_if_exists=None if args.force else lambda m: True)
            if ws:
                SessionManager.set_active_workspace_path(path)
                self.app._output({"status": "success", "message": f"Created workspace at {args.path}", "path": str(ws.base_dir)})
            else:
                self.app._error(f"Failed to create workspace at {args.path}")
        elif args.subcommand == "delete":
            path = Path(args.path)
            deleted = Workspace.delete(path, warn_if_exists=None if args.force else lambda m: True)
            if deleted:
                active_path = SessionManager.get_active_workspace_path()
                if active_path and active_path.resolve() == path.resolve():
                    SessionManager.clear_active_workspace()
                self.app._output({"status": "success", "message": f"Deleted workspace at {args.path}"})
            else:
                self.app._error(f"Failed to delete workspace at {args.path}")
        elif args.subcommand == "info":
            ws = self._get_workspace()
            self.app._output({
                "status": "success",
                "workspace": {
                    "path": str(ws.base_dir),
                    "projects": list(ws.projects._projects.keys()),
                    "datasets": list(ws.datasets._datasets.keys())
                }
            })

class ProjectHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'project'. Use 'list', 'create', 'inspect', or 'delete'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "list":
            self.app._output({"status": "success", "projects": list(ws.projects._projects.keys())})
        elif args.subcommand == "create":
            project = ws.create_project(
                args.name,
                body_parts=args.body_parts.split(",") if args.body_parts else [],
                identities=args.identities.split(",") if args.identities else [],
                behaviors=args.behaviors.split(",") if args.behaviors else [],
            )
            self.app._output({"status": "success", "message": f"Created project {args.name}"})
        elif args.subcommand == "inspect":
            project = ws.projects.get(args.name)
            if not project:
                self.app._error(f"Project '{args.name}' not found")
            self.app._output({"status": "success", "project": project.to_dict() if hasattr(project, "to_dict") else str(project)})
        elif args.subcommand == "delete":
            ws.projects.remove(args.name)
            self.app._output({"status": "success", "message": f"Deleted project {args.name}"})

class DatasetHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'dataset'. Use 'list', 'import', 'inspect', or 'delete'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "list":
            self.app._output({"status": "success", "datasets": list(ws.datasets._datasets.keys())})
        elif args.subcommand == "import":
            ws.create_dataset(args.name, DatasetType(args.type), Path(args.path))
            self.app._output({"status": "success", "message": f"Imported dataset {args.name}"})
        elif args.subcommand == "inspect":
            dataset = ws.get_dataset(args.name)
            if not dataset:
                self.app._error(f"Dataset '{args.name}' not found")
            self.app._output({"status": "success", "dataset": {
                "name": dataset.name,
                "type": dataset.type.value,
                "base_data_path": str(dataset.base_data_path),
                "num_files": len(dataset.files)
            }})
        elif args.subcommand == "delete":
            ws.delete_dataset(args.name)
            self.app._output({"status": "success", "message": f"Deleted dataset {args.name}"})

class RunHandler(BaseHandler):
    def handle(self, args: argparse.Namespace, unknown: list[str]):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'run'. Use 'list', 'schema', or 'exec'.")
            return

        if args.subcommand == "list":
            jobs = JobRegistry.list_jobs()
            output = {name: {"description": schema.description} for name, schema in jobs.items()}
            self.app._output(output)
        elif args.subcommand == "schema":
            schema = JobRegistry.get_job_schema(args.job_name)
            if not schema:
                self.app._error(f"Job '{args.job_name}' not found", code=2)
            self.app._output(schema.params_type.model_json_schema())
        elif args.subcommand == "exec":
            ws = self._get_workspace()
            job_class = JobRegistry.get_job_class(args.job_name)
            job_schema = JobRegistry.get_job_schema(args.job_name)
            if not job_class or not job_schema:
                self.app._error(f"Job '{args.job_name}' not found", code=2)
            
            params_dict = {}
            if args.params:
                params_dict = json.loads(args.params)
            elif hasattr(args, "params_path") and args.params_path:
                path = Path(args.params_path)
                if not path.exists():
                    self.app._error(f"Parameters file not found: {args.params_path}", code=2)
                try:
                    with open(path, "r") as f:
                        params_dict = json.load(f)
                except Exception as e:
                    self.app._error(f"Failed to parse parameters file: {e}", code=2)
            
            i = 0
            while i < len(unknown):
                arg = unknown[i]
                if arg.startswith("--"):
                    key = arg[2:]
                    if i + 1 < len(unknown) and not unknown[i+1].startswith("--"):
                        val = unknown[i+1]
                        if val.lower() == "true": val = True
                        elif val.lower() == "false": val = False
                        else:
                            try:
                                if "." in val: val = float(val)
                                else: val = int(val)
                            except ValueError:
                                pass
                        params_dict[key] = val
                        i += 2
                    else:
                        params_dict[key] = True
                        i += 1
                else:
                    i += 1
            
            try:
                params = job_schema.params_type.model_validate(params_dict)
            except Exception as e:
                self.app._error(f"Validation error: {e}", code=2)
            
            job = job_class(workspace=ws, params=params)
            
            try:
                result = job.run()
                self.app._output({"status": "success", "job": args.job_name, "result": result})
            except Exception as e:
                self.app._error(f"Job failed: {e}")

class SystemHandler(BaseHandler):
    def version(self):
        self.app._output({"version": VERSION_NUMBER, "app": "WHISKER"})
    
    def status(self, job_id: Optional[str]):
        self.app._output({"status": "no background jobs supported in synchronous mode", "job_id": job_id})
        
    def inspect(self, obj_id: str):
        ws = self._get_workspace()
        
        project = ws.projects.get(obj_id)
        if project:
            self.app._output({"type": "project", "data": project.to_dict() if hasattr(project, "to_dict") else str(project)})
            return

        dataset = ws.get_dataset(obj_id)
        if dataset:
            self.app._output({"type": "dataset", "data": {
                "name": dataset.name,
                "type": dataset.type.value,
                "base_data_path": str(dataset.base_data_path),
                "num_files": len(dataset.files)
            }})
            return

        self.app._error(f"Object '{obj_id}' not found in current workspace", code=4)

class PoseEvalHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'pose-eval'. Use 'list' or 'get'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "list":
            evals = {}
            for model_name, datasets in ws.pose_predictions._pose_predictions.items():
                for dataset_name in datasets:
                    metrics = ws.pose_predictions.get_evaluation_metrics(model_name, dataset_name)
                    if metrics:
                        evals.setdefault(model_name, []).append(dataset_name)
            self.app._output({"status": "success", "evaluations": evals})
        elif args.subcommand == "get":
            metrics = ws.pose_predictions.get_evaluation_metrics(args.run_name, args.dataset_name)
            if metrics:
                self.app._output({
                    "status": "success", 
                    "run_name": args.run_name, 
                    "dataset_name": args.dataset_name, 
                    "metrics": metrics
                })
            else:
                self.app._error(f"No evaluation results found for run '{args.run_name}' and dataset '{args.dataset_name}'")

class PoseModelHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'pose-model'. Use 'list' or 'inspect'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "list":
            self.app._output({"status": "success", "models": ws.pose_models.get()})
        elif args.subcommand == "inspect":
            config = ws.pose_models.get_training_config(args.model_name)
            if not config:
                self.app._error(f"Model '{args.model_name}' not found or missing metadata")
            
            history = ws.pose_models.get_training_history(args.model_name)
            split_info = ws.pose_models.get_split_info(args.model_name)
            
            train_size = 0
            val_size = 0
            if split_info and "split_data" in split_info:
                train_size = len(split_info["split_data"].get("train", []))
                val_size = len(split_info["split_data"].get("val", []))
            
            self.app._output({
                "status": "success",
                "model_name": args.model_name,
                "training_config": config,
                "training_history": history,
                "dataset_size": {
                    "train": train_size,
                    "val": val_size,
                    "total": train_size + val_size
                }
            })

class PoseTrainingHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'pose-training'. Use 'run'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "run":
            from whisker.services.pose_estimation.public.topics import PoseTrainingParams
            from whisker.services.pose_estimation.internal.workers.pose_training import PoseTrainingJob
            
            params = PoseTrainingParams(
                run_name=args.run_name,
                dataset_names=args.datasets.split(","),
                individuals=args.individuals.split(","),
                bodyparts=args.bodyparts.split(","),
                backend=args.backend,
                backend_params={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
                },
                resume_run_name=args.resume
            )
            
            job = PoseTrainingJob(workspace=ws, params=params)
            
            self.app._output({"status": "starting", "job": "pose_training", "params": params.model_dump()})
            
            try:
                result = job.run()
                self.app._output({"status": "success", "result": result})
            except Exception as e:
                self.app._error(f"Training failed: {e}")

class RoiTemplateHandler(BaseHandler):
    def handle(self, args: argparse.Namespace):
        if not args.subcommand:
            self.app._error("Missing subcommand for 'roi-template'. Use 'get' or 'set'.")
            return

        ws = self._get_workspace()
        if args.subcommand == "get":
            template = ws.roi_labels.get_template(args.dataset_name)
            if template is None:
                self.app._error(f"No template found for dataset '{args.dataset_name}'", code=4)
            self.app._output(template)
            
        elif args.subcommand == "set":
            path = Path(args.template_path)
            if not path.exists():
                self.app._error(f"Template file not found: {args.template_path}", code=2)
            try:
                with open(path, "r") as f:
                    template_dict = json.load(f)
            except Exception as e:
                self.app._error(f"Failed to parse template file: {e}", code=2)
                
            try:
                ws.create_arena_project(args.dataset_name, template_dict)
                self.app._output({"status": "success", "message": f"Successfully set ROI template and created virtual crop clips for dataset '{args.dataset_name}'"})
            except Exception as e:
                self.app._error(f"Failed to apply ROI template: {e}")

