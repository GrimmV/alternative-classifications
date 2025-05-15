from llm_models.ResponseModels.StatementRefactoringResponse import StatementRefactoringResponse
from typing import Type
from llm_models.llama_model import LlamaModel

class StatementRefactorer(LlamaModel[StatementRefactoringResponse]):

    def get_response_model(self) -> Type[StatementRefactoringResponse]:
        return StatementRefactoringResponse
    
    

