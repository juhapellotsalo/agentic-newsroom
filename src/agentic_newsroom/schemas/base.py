import re
from pathlib import Path
from pydantic import BaseModel


def get_project_root() -> Path:
    """Get the project root directory.
    
    Assumes this file is at src/agentic_newsroom/schemas/base.py
    and project root is 4 levels up.
    """
    return Path(__file__).parent.parent.parent.parent


class NewsroomModel(BaseModel):
    """Base with serialization methods"""

    @classmethod
    def _get_model_name(cls) -> str:
        """Generate snake_case name from class name"""
        class_name = cls.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()

    @classmethod
    def get_serialization_path(cls, slug: str) -> Path:
        serialization_path = get_project_root() / "artifacts" / slug
        if not serialization_path.exists():
            serialization_path.mkdir(parents=True, exist_ok=True)
        return serialization_path

    def save_json(self, slug: str):
        json_obj = self.model_dump_json(indent=2)
        output_dir = self.get_serialization_path(slug)
        name = self._get_model_name()
        with open(output_dir / f"{name}.json", "w") as f:
            f.write(json_obj)

    def save_markdown(self, slug: str):
        markdown_obj = self.to_markdown()
        output_dir = self.get_serialization_path(slug)
        name = self._get_model_name()
        with open(output_dir / f"{name}.md", "w") as f:
            f.write(markdown_obj)


    def to_markdown(self) -> str:
        raise NotImplementedError("Subclasses must implement to_markdown")


    def save(self, slug: str):
        """Save both JSON and Markdown versions."""
        self.save_json(slug)
        self.save_markdown(slug)

    @classmethod
    def load(cls, slug: str):
        """Load instance from JSON file using slug."""
        output_dir = cls.get_serialization_path(slug)
        name = cls._get_model_name()
        json_path = output_dir / f"{name}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"No saved {cls.__name__} found for slug '{slug}' at {json_path}")
            
        with open(json_path, "r") as f:
            json_content = f.read()
            
        return cls.model_validate_json(json_content)
                

    
