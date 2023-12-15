from dataclasses import dataclass
import datetime
import torch


@dataclass
class Section:
    id: str
    content: str
    created: datetime.datetime
    last_modified: datetime.datetime
    embedding: torch.vector
    key_phrases: list[str]

    def edit(self, new_content):
        self.content = new_content


@dataclass
class Document:
    sections: list[Section]
    last_modified: datetime.datetime

    def writing_order():
        pass


    def temporal_order():
        pass