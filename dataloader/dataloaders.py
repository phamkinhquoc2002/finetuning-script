import os
from typing import Optional, Union
from utils.logger import log_message
from utils.trainers import conversation_format, standard_format, preference_format
from datasets import Dataset, load_dataset
from prompts import system_prompt

class BaseDataLoader():

    def __init__(self,
                 path: str,
                 format: Optional[str]):
        self.path = path
        self.format = format

    def load(self):
        raise NotImplementedError

    def pre_process(self, ) -> Union[callable, str]:
        if self.format == "no":
            return "no"
        elif self.format == "conversational":
            return lambda sample: conversation_format(system_prompt, sample)
        elif self.format == "standard":
            return lambda sample: standard_format(sample)
        elif self.format == "preference":
            return lambda sample: preference_format(sample)

class CSVDataLoader(BaseDataLoader):

    def __init__(self,
                 path: str,
                 format: Optional[str]):
        super().__init__(path, format)
        if not path.endswith('.csv') or not os.path.exists(path):
            log_message(
                {
                    "type":"ERROR",
                    "text": "Cant locate any CSV Files in the path you provided!"
                }
            )

    def load(self) -> Optional[Dataset]:
        try:
            dataset = load_dataset('csv', data_files=self.path, split='train')
        except Exception as e:
            log_message(
                {
                    "type": "ERROR", 
                    "text": f"Failed to load dataset: {e}"
                    }
                )
            raise
        pre_processing = self.pre_process()
        if pre_processing == "no":
            return dataset
        return dataset.map(pre_processing, remove_columns=dataset.column_names)
    
class JSONDataLoader(BaseDataLoader):

    def __init__(self, path: str, format: Optional[str]):
        super().__init__(path, format)
        if not path.endswith('.json') or not os.path.exists(path):
            log_message(
                {
                    "type": "ERROR", 
                    "text": "Invalid or missing JSON file path!"
                    }
                    )

    def load(self) -> Optional[Dataset]:
        try:
            dataset = load_dataset('json', data_files=self.path, split='train')
        except Exception as e:
            log_message(
                {
                    "type": "ERROR", 
                    "text": f"Failed to load dataset: {e}"
                 }
                )
            return None

        pre_processing = self.pre_process()
        if not pre_processing:
            return dataset

        return dataset.map(pre_processing, remove_columns=dataset.column_names)

class HuggingFaceDataLoader(BaseDataLoader):

    def __init__(self, path: str, format: Optional[str]):
        super().__init__(path, format)

    def load(self) -> Optional[Dataset]:
        try:
            dataset = load_dataset(self.path, split='train')
        except Exception as e:
            log_message({"type": "ERROR", "text": f"Failed to load dataset from Hugging Face: {e}"})
            return None

        pre_processing = self.pre_process()
        if not pre_processing:
            return dataset

        return dataset.map(pre_processing, remove_columns=dataset.column_names)