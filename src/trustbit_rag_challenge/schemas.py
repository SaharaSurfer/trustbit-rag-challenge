from typing import Any, Literal

from pydantic import BaseModel, Field

from trustbit_rag_challenge.enums import QuestionKind

# ----------------------------------- Mixins -----------------------------------


class ChainOfThoughtMixin(BaseModel):
    step_by_step_analysis: str = Field(
        ...,
        description=(
            "Detailed step-by-step analysis of the answer with at least 5 "
            "steps and at least 150 words. Pay special attention to the "
            "wording of the question to avoid being tricked. Sometimes it "
            "seems that there is an answer in the context, but this might "
            "be not the requested value, but only a similar one. If the "
            "metric requires calculation or aggregation not present in the "
            "text, explicitely state that."
        ),
    )
    reasoning_summary: str = Field(
        ...,
        description=(
            "Concise summary of the step-by-step reasoning process. "
            "Around 50 words."
        ),
    )


class ReferencesMixin(BaseModel):
    relevant_pages: list[int] = Field(
        ...,
        description=(
            "List of page numbers containing information directly used to "
            "answer the question. Include only:\n"
            "- Pages with direct answers or explicit statements\n"
            "- Pages with key information that strongly supports the answer\n"
            "Do not include pages with only tangentially related information "
            "or weak connections to the answer. At least one page should "
            "be included in the list. "
        ),
    )


# ----------------------------- LLM Response Models ----------------------------


class NumberResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: float | int | Literal["N/A"] = Field(
        ...,
        description=(
            "The exact numeric value based on the text is required. "
            "Mandatory formatting rules:\n\n"
            "- Example for commas:\n"
            "Value from context: 197,621,000\n"
            "Final answer: 197621000\n\n"
            "- Example for percentages:\n"
            "Value from context: 58,3%\n"
            "Final answer: 58.3\n\n"
            "- Example for negative values:\n"
            "Value from context: (2,124,837) CHF\n"
            "Final answer: -2124837\n\n"
            "- Example for numbers in thousands:\n"
            "Value from context: 4970,5 (in thousands €)\n"
            "Final answer: 4970500\n\n"
            "- Example for numbers in millions:\n"
            "Value from context: 20.7 (in millions $)\n"
            "Final answer: 20700000\n\n"
            "- Example for numbers in billions:\n"
            "Value from context: 6.2 (in billions $)\n"
            "Final answer: 6200000000\n\n"
            "- Example for currency mismatch:\n"
            "Value from context: 780000 USD, but question mentions EUR\n"
            "Final answer: 'N/A'\n\n"
            "- Example for approximation:\n"
            "Value from context: 'over 500'\n"
            "Final answer: 'N/A'\n\n"
            "- Example for combined scope:\n"
            "Value from context: '500 patents and applications'\n"
            "Question asks for: 'patents'\n"
            "Final answer: 'N/A'\n\n"
            "- Return 'N/A' if metric is not directly stated EVEN IF it could "
            "be calculated from other metrics in the context\n\n"
            "- Return 'N/A' if EXACT metric is not available in the context"
        ),
    )


class BooleanResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: bool = Field(
        ...,
        description=(
            "True or False based on EXPLICIT statements. "
            "- Return True ONLY if the text explicitly states the event/change "
            "occurred.\n"
            "- Return False if the answer requires inferring a consequence.\n"
            "- Return False if the text is silent or ambiguous.\n"
            "- Return False if the text says X did NOT happen."
        ),
    )


class NameResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: str | Literal["N/A"] = Field(
        ...,
        description=(
            "The specific name of the entity/person/product found. "
            "Extract exactly as it appears in context without any generic "
            "description. Return 'N/A' if entity is not explicitly named."
        ),
    )


class NamesResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: list[str] | Literal["N/A"] = Field(
        ...,
        description=(
            "A list of names extracted exactly as they appear. "
            "If asking for positions, return ONLY titles (e.g. 'CEO'). "
            "If asking for names, return full names. "
            "No duplicates."
        ),
    )


class ComparativeResponse(ChainOfThoughtMixin):
    final_answer: str | Literal["N/A"] = Field(
        ...,
        description=(
            "The winning company name exactly as in the question, or 'N/A'."
        ),
    )


class RephrasedQuestion(BaseModel):
    company_name: str = Field(
        description="Company name, exactly as provided in the original question"
    )
    question: str = Field(
        description="Rephrased question specific to this company"
    )


class RephrasedQuestions(BaseModel):
    questions: list[RephrasedQuestion] = Field(
        description="List of rephrased questions for each company"
    )


# ------------------------------------ DTOs ------------------------------------


class ClientResponse(BaseModel):
    value: str | Literal["N/A"] | float | int | bool | list[str]
    relevant_pages: list[int] = Field(default_factory=list)
    step_by_step_analysis: str = ""
    reasoning_summary: str = ""


class RouterResponse(BaseModel):
    value: str | Literal["N/A"] | float | int | bool | list[str]
    references: list[dict[str, Any]] = Field(default_factory=list)


# ------------------------------ Submission Models -----------------------------


class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(
        ..., description="Zero-based physical page number in the PDF file"
    )


class Answer(BaseModel):
    question_text: str | None = Field(None, description="Text of the question")
    kind: QuestionKind | None = Field(None, description="Kind of the question")
    value: float | str | bool | list[str] | Literal["N/A"] = Field(
        ..., description="Answer to the question"
    )
    references: list[SourceReference] = Field(
        [], description="References to the source material"
    )


class AnswerSubmission(BaseModel):
    team_email: str = Field(
        ..., description="Email that your team used to register"
    )
    submission_name: str = Field(
        ..., description="Unique name of the submission"
    )
    answers: list[Answer] = Field(..., description="List of answers")
