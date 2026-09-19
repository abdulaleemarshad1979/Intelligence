"""Suspect gallery manager loading known records from repository."""

from typing import List, Optional
from app.database.models import CriminalRecord
from app.database.repository import Repository

class SuspectGallery:
    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.suspects: List[CriminalRecord] = []
        self.reload()

    def reload(self):
        self.suspects = self.repo.get_all_criminal_records()

    def get_all(self) -> List[CriminalRecord]:
        if not self.suspects:
            self.reload()
        return self.suspects

    def get_all_suspects(self) -> List[CriminalRecord]:
        return self.get_all()

    def find_by_id(self, suspect_id: str) -> Optional[CriminalRecord]:
        for s in self.suspects:
            if s.id == suspect_id:
                return s
        return self.repo.get_criminal_record_by_id(suspect_id)
