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

class RoleplayingDistributionShift(DialogueDataset):
    base_name = "roleplaying_distribution_shift"
    padding = {model_type: {"left": 0, "right": 0} for model_type in ["gemma", "mistral", "llama"]}
    
    all_variants: list[str] = [
        "plain", 
        "subset",
        "french",
        "greek"
    ]
    
    def _get_dialogues(self) -> DialogueDataType:
        statements, labels = self._load_data()
        dialogues = self._get_prefixed_dialogues(statements, labels)
        
        labels_new = [Label.DECEPTIVE if label == 1 else Label.HONEST for label in labels]
        return dialogues, labels_new, None
        
    def _load_data(self) -> tuple[list[dict], list[int]]:
        if self.variant == "plain":
            data_path: Path = REPO_ROOT / "data/roleplaying_distr_shift/roleplaying_clean.json"
        elif self.variant == "subset":
            data_path: Path = REPO_ROOT / "data/roleplaying_distr_shift/roleplaying_subset.json"
        elif self.variant == "french":
            data_path: Path = REPO_ROOT / "data/roleplaying_distr_shift/roleplaying_french.json"
        elif self.variant == "greek":
            data_path: Path = REPO_ROOT / "data/roleplaying_distr_shift/roleplaying_greek.json"
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        with open(data_path, "r") as f:
            data = json.load(f)

        statements = []
        labels = []
        
        for item in data:
            statements.append(item)
            labels.append(item['label'])
            
        return statements, labels
    
    def _get_prefixed_dialogues(
        self, 
        statements: list[dict],
        labels: list[int],
    ) -> list[Dialogue]:
        
        dialogues: list[Dialogue] = []
        for statement in statements:
            dialogue: Dialogue = []
            dialogue.append(
                Message(
                    "system",
                    statement["system_message"],
                    False
                )
            )

            dialogue.append(
                Message(
                    "user",
                    statement["user_message"],
                    False
                )
            )

            dialogue.append(
                Message(
                    "assistant",
                    statement["answer_prefix"],
                    False
                )
            )
            
            dialogue.append(
                Message(
                    "assistant",
                    statement["assistant_response"],
                    True
                )
            )
            
            dialogues.append(dialogue)
            
        return dialogues