import importlib
import importlib.util
from pathlib import Path
from typing import Type

from projects.base_project import BaseProject
from shared.config import config


class ProjectLoader:
    """
    Dynamically discovers and loads projects from the projects directory.

    Projects are discovered by looking for directories containing a project.py
    file with a class that inherits from BaseProject.
    """

    def __init__(self):
        self.projects: dict[str, Type[BaseProject]] = {}
        self._project_instances: dict[str, BaseProject] = {}

    def discover_projects(self) -> list[dict]:
        """
        Scan the projects directory and discover all available projects.

        Returns:
            List of project info dicts
        """
        self.projects.clear()
        project_infos = []

        projects_dir = config.projects_dir

        for item in projects_dir.iterdir():
            # Skip non-directories and special files
            if not item.is_dir() or item.name.startswith('_'):
                continue

            # Skip if no project.py file
            project_file = item / "project.py"
            if not project_file.exists():
                continue

            try:
                # Load the module
                project_class = self._load_project_class(item.name, project_file)
                if project_class:
                    self.projects[item.name] = project_class

                    # Get project info without instantiating
                    info = {
                        "id": item.name,
                        "name": getattr(project_class, 'name', item.name),
                        "description": getattr(project_class, 'description', ''),
                        "author": getattr(project_class, 'author', 'Unknown'),
                        "version": getattr(project_class, 'version', '1.0.0'),
                    }
                    project_infos.append(info)
                    print(f"Discovered project: {info['name']}")

            except Exception as e:
                print(f"Error loading project from {item.name}: {e}")

        return project_infos

    def _load_project_class(
        self, project_id: str, project_file: Path
    ) -> Type[BaseProject] | None:
        """Load a project class from a project.py file."""
        spec = importlib.util.spec_from_file_location(
            f"projects.{project_id}.project",
            project_file
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the BaseProject subclass in the module
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseProject)
                and obj is not BaseProject
            ):
                return obj

        return None

    def get_project(self, project_id: str) -> BaseProject | None:
        """
        Get or create an instance of a project.

        Args:
            project_id: The project directory name

        Returns:
            Project instance or None if not found
        """
        if project_id in self._project_instances:
            return self._project_instances[project_id]

        if project_id not in self.projects:
            return None

        try:
            instance = self.projects[project_id]()
            self._project_instances[project_id] = instance
            return instance
        except Exception as e:
            print(f"Error instantiating project {project_id}: {e}")
            return None

    def unload_project(self, project_id: str):
        """Unload a project instance to free resources."""
        if project_id in self._project_instances:
            try:
                self._project_instances[project_id].stop()
            except Exception as e:
                print(f"Error stopping project {project_id}: {e}")
            del self._project_instances[project_id]

    def list_projects(self) -> list[str]:
        """Get list of discovered project IDs."""
        return list(self.projects.keys())
