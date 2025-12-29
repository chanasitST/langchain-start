import json
import os
from typing import List

class SemanticMemory:
    # A simple semantic memory store that saves facts into a JSON file
    # In real world use cases, you might wanna use db or vector db

    def __init__(self, file_path: str = "user_profile.json"):
        self.file_path = file_path
        self._load_memory()

    def _load_memory(self):
        # Loads facts from json file
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.facts = json.load(f)
            except json.JSONDecodeError:
                self.facts = []
        else:
            self.facts = []

    def _save_memory(self):
        # Saves Facts into the json files
        # Ensure dir exists, create if not
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.facts, f, indent=2)
        
    def save_fact(self, fact: str):
        # Saves a new fact if it doesn't exist
        if fact not in self.facts:
            self.facts.append(fact)
            self._save_memory()
            print(f"Saved to Semantic Memory: {fact}")

        else:
            print(f"Fact already exists in Semantic Memory: {fact}")

    def get_all_facts(self) -> List[str]:
        # return all stored facts
        return self.facts
    
    def get_relevant_facts(self, query:str) -> List[str]:
        # return relevant facts (simple return all) because no vector sim

        return self.facts