import ast
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Optional

# Configure logging to look clean and professional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ArchLinter")


class ArchitectureRule(NamedTuple):
    target_dir_name: str                           # The directory being inspected (e.g., "public", "core_service")
    forbidden_keyword: str                         # The module name that should never be imported inside it
    path_exclusions: Optional[list[str]] = None    # Skip rule if this substring is in the file path
    import_exclusions: Optional[list[str]] = None  # Skip violation if this substring is in the imported module


# =====================================================================
# CENTRALIZED CONFIGURATION: Define your architectural firewalls here!
# =====================================================================
ARCHITECTURE_RULES = [
    ArchitectureRule(target_dir_name="public", forbidden_keyword="internal"),

    ArchitectureRule(target_dir_name="base", forbidden_keyword="gui", path_exclusions=["gui"]),
    ArchitectureRule(target_dir_name="base", forbidden_keyword="services", path_exclusions=["gui"]),

    ArchitectureRule(target_dir_name="core", forbidden_keyword="gui"),
    # ArchitectureRule(
    #    target_dir_name="core",
    #    forbidden_keyword="services",
    #    path_exclusions=["services"],
    #    import_exclusions=["public"]
    # ),

    ArchitectureRule(target_dir_name="animal_detection", forbidden_keyword="pose_estimation"),
    ArchitectureRule(target_dir_name="pose_estimation", forbidden_keyword="behavior_classification"),
    
]


class ImportFinder(ast.NodeVisitor):
    """AST Viewer to extract all imports, resolving relative targets safely."""
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)


def get_all_imports_from_file(file_path: Path) -> list[str]:
    """Parses Python source file via AST and returns all import targets."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        finder = ImportFinder()
        finder.visit(tree)
        return finder.imports
    except SyntaxError:
        logger.warning(f"Skipping unparseable syntax in: {file_path.name}")
        return []


def run_import_linter(project_root: Path) -> bool:
    """
    Scans the project for structural violations based on the rules constant.
    """
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        return False

    violations = 0
    scanned_count = 0

    logger.info(f"Starting architectural linting across: {project_root.resolve()}")
    logger.info(f"Enforcing {len(ARCHITECTURE_RULES)} strict separation boundaries...")

    for file_path in project_root.glob("**/*.py"):
        parts = file_path.parts
        imports = []
        
        for rule in ARCHITECTURE_RULES:
            if rule.target_dir_name in parts:
                # 1. Path-based exclusion check (e.g., skips files inside a specific folder)
                
                if rule.path_exclusions:
                    matched = False
                    for path in rule.path_exclusions:
                        if path in parts:
                            matched=True
                            break
                    if matched:
                        continue

                if not imports:
                    imports = get_all_imports_from_file(file_path)
                    scanned_count += 1
                
                relative_display_path = file_path.relative_to(project_root)
                
                for imp in imports:
                    if rule.forbidden_keyword in imp:
                        # 2. Import-string-based exclusion check (e.g., allows explicit utility/mock submodules)
                        if rule.import_exclusions:
                            matched=False
                            for exclusion in rule.import_exclusions:
                                if exclusion in imp:
                                    matched=True
                                    break
                                
                            if matched:
                                continue

                        logger.error(
                            f"❌ VIOLATION in [{relative_display_path}]: "
                            f"Layer '{rule.target_dir_name}' is pulling in forbidden module -> '{imp}' "
                            f"(Rule: No '{rule.forbidden_keyword}')"
                        )
                        violations += 1

    logger.info("--------------------------------------------------")
    logger.info(f"Scan complete. Total files evaluated against rules: {scanned_count}")
    
    if violations > 0:
        logger.error(f"💥 FAILURE: Found {violations} dependency rule violations!")
        return False
        
    logger.info("✅ SUCCESS: All architectural boundaries are perfectly intact.")
    return True


if __name__ == "__main__":
    DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
    scan_target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECT_ROOT

    success = run_import_linter(scan_target)
    sys.exit(0 if success else 1)