from typing import Literal

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
            "seems that there is an answer in the context, but this is might "
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
            "An exact metric number is expected.\n"
            "- For percentages (58.3%) -> 58.3\n"
            "- For negative values ((2,124)) -> -2124\n"
            "- For values in thousands (4,970.5 in thousands) -> 4970500\n"
            "- If currency differs from question -> 'N/A'\n"
            "- If not directly stated (requires calc) -> 'N/A'"
        ),
    )


class BooleanResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: bool = Field(
        ...,
        description=(
            "True or False. If the question asks 'Did X happen?' and the "
            "text says X happened -> True. If the text says X did NOT "
            "happen -> False. If the text mentions nothing about X -> False."
        ),
    )


class NameResponse(ChainOfThoughtMixin, ReferencesMixin):
    final_answer: str | Literal["N/A"] = Field(
        ...,
        description=(
            "The specific name of the entity/person/product found. "
            "Extract exactly as it appears in context. "
            "Return 'N/A' if not available."
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


# -------------------------------- Helper Models -------------------------------


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
