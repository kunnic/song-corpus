import os

from abc import (
    ABC, 
    abstractmethod
)
from typing import Any, Dict, List, Optional

class DataLoader(ABC):
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path: str = file_path
        self._data: Any = None

    @abstractmethod
    def load(self) -> Any:
        pass

    @property
    def data(self) -> Any:
        return self._data