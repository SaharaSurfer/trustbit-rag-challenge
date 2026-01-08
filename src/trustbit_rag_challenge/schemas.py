from typing import Any, Literal

from pydantic import BaseModel, Field

from trustbit_rag_challenge.enums import QuestionKind

# ----------------------------------- Mixins -----------------------------------


class ChainOfThoughtMixin(BaseModel):
    """
    A mixin to force LLM generate a step-by-step analysis
    before providing the final answer.

    Attributes
    ----------
    step_by_step_analysis : str
        A detailed breakdown of the logic used to derive the answer.
    reasoning_summary : str
        A summary of the reasoning.
    """

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
    """
    A mixin to enforce source citation.

    Used by response models that extract data directly from document chunks.

    Attributes
    ----------
    relevant_pages : list[int]
        A list of integer page numbers from the provided context that contain
        the direct evidence for the answer.
    """

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
    """
    Schema for questions requiring a numerical answer.

    Attributes
    ----------
    final_answer : float | int | Literal["N/A"]
        The extracted number or 'N/A' if the data is missing or ambiguous.
    """

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
    """
    Schema for questions requiring a True/False answer.

    Attributes
    ----------
    final_answer : bool
        True if the condition is met based on the text, False otherwise.
    """

    final_answer: bool = Field(
        ...,
        description=(
            "True or False. If the question asks 'Did X happen?' and the "
            "text says X happened -> True. If the text says X did NOT "
            "happen -> False. If the text mentions nothing about X -> False."
        ),
    )


class NameResponse(ChainOfThoughtMixin, ReferencesMixin):
    """
    Schema for questions requiring a specific entity name.

    Attributes
    ----------
    final_answer : str | Literal["N/A"]
        The extracted name (e.g., person, company, product) or 'N/A'.
    """

    final_answer: str | Literal["N/A"] = Field(
        ...,
        description=(
            "The specific name of the entity/person/product found. "
            "Extract exactly as it appears in context. "
            "Return 'N/A' if not available."
        ),
    )


class NamesResponse(ChainOfThoughtMixin, ReferencesMixin):
    """
    Schema for questions requiring a list of entities.

    Attributes
    ----------
    final_answer : list[str] | Literal["N/A"]
        A list of extracted names/titles or 'N/A' if none found.
    """

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
    """
    Schema for comparative questions.

    Inherits `ChainOfThoughtMixin` to explain the comparison logic.
    Does NOT inherit `ReferencesMixin` because this model operates on
    aggregated summaries, not directly on PDF pages.

    Attributes
    ----------
    final_answer : str | Literal["N/A"]
        The winning company name or 'N/A'.
    """

    final_answer: str | Literal["N/A"] = Field(
        ...,
        description=(
            "The winning company name exactly as in the question, or 'N/A'."
        ),
    )


class RephrasedQuestion(BaseModel):
    """
    Schema for a single decomposed question targeting specific company.

    Used during the processing of comparative questions to isolate the query
    logic for one specific entity before aggregation.

    Attributes
    ----------
    company_name : str
        The name of the specific company extracted from the original
        comparative question (e.g., "Apple").
    question : str
        The standalone, rephrased question targeted specifically at
        this company (e.g., "What was Apple's revenue in 2022?"), removing
        references to other entities.
    """

    company_name: str = Field(
        description="Company name, exactly as provided in the original question"
    )
    question: str = Field(
        description="Rephrased question specific to this company"
    )


class RephrasedQuestions(BaseModel):
    """
    Schema for the output of the query decomposition step.

    This container holds the list of individual questions generated by the LLM
    when breaking down a complex comparative query.

    Attributes
    ----------
    questions : list[RephrasedQuestion]
        A list of individual question objects, one for each company
        mentioned in the original query.
    """

    questions: list[RephrasedQuestion] = Field(
        description="List of rephrased questions for each company"
    )


# ------------------------------------ DTOs ------------------------------------


class ClientResponse(BaseModel):
    """
    Internal DTO returned by the LLMClient.

    Attributes
    ----------
    value : Union[str, float, int, bool, list[str]]
        The extracted answer payload.
    relevant_pages : list[int]
        List of page indices referenced by the LLM.
    step_by_step_analysis : str
        The full reasoning chain.
    reasoning_summary : str
        A brief summary of the answer justification.
    """

    value: str | Literal["N/A"] | float | int | bool | list[str]
    relevant_pages: list[int] = Field(default_factory=list)
    step_by_step_analysis: str = ""
    reasoning_summary: str = ""


class RouterResponse(BaseModel):
    """
    Internal DTO returned by the RAGRouter.

    Attributes
    ----------
    value : Union[str, float, int, bool, list[str]]
        The final answer value.
    references : list[dict[str, Any]]
        A list of dictionaries containing `pdf_sha1` and `page_index`.
    """

    value: str | Literal["N/A"] | float | int | bool | list[str]
    references: list[dict[str, Any]] = Field(default_factory=list)


# ------------------------------ Submission Models -----------------------------


class SourceReference(BaseModel):
    """
    Schema for a single reference in the final submission.

    Attributes
    ----------
    pdf_sha1 : str
        The SHA1 hash of the source PDF document.
    page_index : int
        The zero-based physical page index in the PDF.
    """

    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(
        ..., description="Zero-based physical page number in the PDF file"
    )


class Answer(BaseModel):
    """
    Schema for a single answer object in the submission file.

    Attributes
    ----------
    question_text : str | None
        The original text of the question.
    kind : QuestionKind | None
        The type of the question.
    value : Union[float, str, bool, list[str], Literal["N/A"]]
        The final answer.
    references : list[SourceReference]
        List of supporting document references.
    """

    question_text: str | None = Field(None, description="Text of the question")
    kind: QuestionKind | None = Field(None, description="Kind of the question")
    value: float | str | bool | list[str] | Literal["N/A"] = Field(
        ..., description="Answer to the question"
    )
    references: list[SourceReference] = Field(
        [], description="References to the source material"
    )


class AnswerSubmission(BaseModel):
    """
    Schema for the submission file.

    Attributes
    ----------
    team_email : str
        Registration email of the team.
    submission_name : str
        Unique identifier for this submission version.
    answers : list[Answer]
        The collection of answers to all challenge questions.
    """

    team_email: str = Field(
        ..., description="Email that your team used to register"
    )
    submission_name: str = Field(
        ..., description="Unique name of the submission"
    )
    answers: list[Answer] = Field(..., description="List of answers")
