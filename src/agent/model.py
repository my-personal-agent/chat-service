from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """Structured final response model"""

    answer: Annotated[str, Field(description="The main answer to the user's question")]
    confidence: Annotated[
        float, Field(description="Confidence score between 0 and 1", ge=0, le=1)
    ]
    sources_used: Annotated[
        List[str],
        Field(description="List of tools/sources used to generate the answer"),
    ]
    reasoning: Annotated[
        str, Field(description="Brief explanation of the reasoning process")
    ]
    follow_up_suggestions: Annotated[
        Optional[List[str]],
        Field(description="Suggested follow-up questions", default=[]),
    ]
