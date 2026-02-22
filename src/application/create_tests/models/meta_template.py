# from abc import ABC
# from typing import Dict
# 
# from langchain_core.prompts import PromptTemplate
# 
# 
# class MetaTemplate(ABC):
# 
#     self.prompt_template: PromptTemplate
#     self.version: int
#     self.name: str
# 
#     def __init__(self, version: int, name: str):
#         self.version = version
#         self.name = name
# 
#     @abstractmethod
#     def get_metada(self) -> Dict:
#         pass
