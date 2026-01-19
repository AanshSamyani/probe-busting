import json

from pathlib import Path
from deception_detection.data.base import (
    Dialogue,
    DialogueDataset,
    DialogueDataType,
    Message,
)
from deception_detection.types import Label
from deception_detection.utils import REPO_ROOT

class HarmPressureChoice(DialogueDataset):
    base_name = "harm_pressure_choice"
    padding = {model_type: {"left": 0, "right": 0} for model_type in ["gemma", "mistral", "llama"]}
    
    all_variants: list[str] = [
        "test"
    ]
    
    def _get_dialogues(self) -> DialogueDataType:
        statements, labels = self._load_data()
        dialogues = self._get_prefixed_dialogues(
            statements,
            labels,
            user_messages=None,
            sys_prefixes=None
        )
        
        labels_new = [Label.DECEPTIVE if label == 1 else Label.HONEST for label in labels]
        
        return dialogues, labels_new, None
        
    def _load_data(self) -> tuple[list[str], list[int]]:
        if self.variant == "test":
            data_path: Path = REPO_ROOT / "data/harm_pressure/harm_pressure_choice.json"
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        with open(data_path, "r") as f:
            data = json.load(f)
        
        statements = []
        labels = []
        for i in range(len(data)):
            statements.append(data[i]["messages"])
            if data[i]["label"] == True:
                labels.append(1)
            else:
                labels.append(0)
                
        return statements, labels
    
    def _get_prefixed_dialogues(
        self,
        statements: list[str],
        labels: list[int],
        user_messages: tuple[str, str] | None,
        sys_prefixes: tuple[str, str] | None,
    ) -> list[Dialogue]:
        
        dialogues: list[Dialogue] = []
        for statement, label in zip(statements, labels):
            dialogue: Dialogue = []
            for msg in statement:
                dialogue.append(Message(role=msg["role"], content=msg["content"], detect=msg["detect"]))
            
            dialogues.append(dialogue)
            
        return dialogues