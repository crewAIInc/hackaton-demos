from pydantic import BaseModel, Field


class ContractClassification(BaseModel):
    category: str = Field(
        description="The classified category of the contract (e.g., 'License Agreement', 'Service', 'IP', etc.)"
    )
    reasoning: str = Field(
        description="A concise explanation (max 150 words) justifying why the contract was classified in this category"
    )
