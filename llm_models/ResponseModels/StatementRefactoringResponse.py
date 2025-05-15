from pydantic import BaseModel, Field

class StatementRefactoringResponse(BaseModel):
    refactored_statement: str = Field(..., description="The refactored statement")
    applied_changes: list[str] = Field(..., description="The changes that were applied to the statement")
