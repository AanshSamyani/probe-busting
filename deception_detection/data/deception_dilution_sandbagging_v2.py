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

class DeceptionDilutionSandbaggingV2Dataset(DialogueDataset):
    base_name = "deception_dilution_sandbagging_v2"
    padding = {model_type: {"left": 0, "right": 0} for model_type in ["gemma", "mistral", "llama"]}
    
    all_variants: list[str] = [
        "plain"
    ] 
    
    def _get_dialogues(self) -> DialogueDataType:
        statements, labels = self._load_data()
        
        dialogues = self._get_prefixed_dialogues(
            statements,
            labels
        )
        
        labels_new = [Label.DECEPTIVE if label == 1 else Label.HONEST for label in labels]
        
        return dialogues, labels_new, None
    
    def _load_data(self) -> tuple[list[str], list[int]]:
        if self.variant == "plain":
            data_path: Path = REPO_ROOT / "data/deception_dilution_sandbagging_v2/deception_dilution_sandbagging_v2_n=10_k=5_d=20_appended.json"
            
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        with open(data_path, "r") as f:
            data = json.load(f)
            
        statements = []
        labels = []
        for key in data.keys():
            for item in data[key]:
                statements.append(item)
                labels.append(1)
                
        return statements, labels
    
    def _get_prefixed_dialogues(
        self,
        statements: list[str],
        labels: list[int],
    ) -> list[Dialogue]:
        dialogues: list[Dialogue] = []
        for statement in statements:
            dialogue: Dialogue = []
            for message in statement:
                dialogue.append(Message(role=message["role"], content=message["content"], detect=message["detect"]))
        
            dialogues.append(dialogue)
            
        return dialogues    
            