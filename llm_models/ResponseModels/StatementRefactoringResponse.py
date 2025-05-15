from pydantic import BaseModel, Field

class StatementRefactoringResponse(BaseModel):
    refactored_statement: str = Field(..., description="The refactored statement")
    refactoring_reason: str = Field(..., description="The reason for the refactoring")
