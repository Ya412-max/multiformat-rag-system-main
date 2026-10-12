from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentMeta:
    source: str = ""
    file_type: str = ""
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    title: str = ""
    extra: dict = field(default_factory=dict)
