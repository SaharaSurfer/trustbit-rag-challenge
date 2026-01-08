import json
import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from trustbit_rag_challenge.config import OPENAI_MODEL
from trustbit_rag_challenge.enums import QuestionKind
from trustbit_rag_challenge.llm.prompts import (
    format_rephrasing_prompt,
    format_user_prompt,
    get_base_system_prompt,
    get_rephrasing_system_prompt,
)
from trustbit_rag_challenge.schemas import (
    BaseModel,
    BooleanResponse,
    ClientResponse,
    ComparativeResponse,
    NameResponse,
    NamesResponse,
    NumberResponse,
    RephrasedQuestion,
    RephrasedQuestions,
)


class LLMClient:
    """
    A wrapper around the OpenAI-compatible API to handle Structured Outputs.

    This client abstracts the complexity of prompt construction, schema
    selection, network retries, and error handling. It ensures that all
    interactions with the LLM result in strictly typed Pydantic objects or
    safe fallback values.

    Attributes
    ----------
    client : OpenAI
        The initialized OpenAI API client.
    model_name : str
        The identifier of the LLM model to use (e.g., 'gpt-4o-mini').
    """

    def __init__(self):
        """
        Initialize the LLM client.

        Loads the API key from environment variables and sets up the
        OpenAI client.

        Raises
        ------
        ValueError
            If 'OPENAI_API_KEY' is not found in the environment variables.
        """

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables"
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = OPENAI_MODEL

    @staticmethod
    def _get_schema_model(kind: QuestionKind) -> type[BaseModel]:
        """
        Select the appropriate Pydantic output schema based on the
        question type.

        Parameters
        ----------
        kind : QuestionKind
            The classification of the question (NUMBER, BOOLEAN, etc.).

        Returns
        -------
        type[BaseModel]
            The Pydantic class (not instance) defining the expected output
            structure.
        """

        mapping: dict[QuestionKind, type[BaseModel]] = {
            QuestionKind.NUMBER: NumberResponse,
            QuestionKind.BOOLEAN: BooleanResponse,
            QuestionKind.NAME: NameResponse,
            QuestionKind.NAMES: NamesResponse,
            QuestionKind.COMPARATIVE: ComparativeResponse,
        }
        return mapping.get(kind, NameResponse)

    @staticmethod
    def _get_fallback_response(
        kind: QuestionKind, error_msg: str
    ) -> ClientResponse:
        """
        Generate fallback response in case of error.

        This ensures the pipeline doesn't crash on a single failed question.
        It returns specific "empty" values based on the question type.

        Parameters
        ----------
        kind : QuestionKind
            The type of question being processed.
        error_msg : str
            Description of the error that occurred.

        Returns
        -------
        ClientResponse
            A DTO containing the fallback value and the error message in the
            reasoning.
        """

        value: str | bool | list[str] = "N/A"
        if kind == QuestionKind.BOOLEAN:
            value = False
        elif kind == QuestionKind.NAMES:
            value = []

        return ClientResponse(
            value=value,
            relevant_pages=[],
            step_by_step_analysis="",
            reasoning_summary=error_msg,
        )

    @staticmethod
    def _log_llm_response(question: str, kind: QuestionKind, result: BaseModel):
        """
        Log the structured response from the LLM for debugging purposes.

        Parameters
        ----------
        question : str
            The user question.
        kind : QuestionKind
            The type of question.
        result : BaseModel
            The Pydantic object returned by the LLM.
        """

        log_payload = {"q": question, "kind": kind, **result.model_dump()}
        logger.info(f"🤖 LLM: {json.dumps(log_payload, ensure_ascii=False)}")

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def _call_llm(
        self, system_instr: str, user_msg: str, response_model: type[BaseModel]
    ) -> Any:
        """
        Execute the raw API call with retries.

        Uses the provider's structured output capabilities to parse the response
        into the given Pydantic model.

        Parameters
        ----------
        system_instr : str
            The system prompt containing instructions and examples.
        user_msg : str
            The user prompt containing the context and the question.
        response_model : type[BaseModel]
            The expected Pydantic schema for the response.

        Returns
        -------
        Any
            The parsed Pydantic object (e.g., NumberResponse, BooleanResponse).
        """
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": system_instr},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            text_format=response_model,
        )
        return response.output_parsed

    def answer_question(
        self, question: str, aggregated_context: str, kind: QuestionKind
    ) -> ClientResponse:
        """
        Process a single question using the provided context.

        1. Selects prompts and schema based on `kind`.
        2. Calls the LLM.
        3. Logs the result.
        4. Converts the specific LLM schema into a unified `ClientResponse` DTO.

        Parameters
        ----------
        question : str
            The question text.
        aggregated_context : str
            Text chunks formatted with page delimiters.
        kind : QuestionKind
            The type of the question.

        Returns
        -------
        ClientResponse
            A unified response object containing the answer, reasoning, and
            page refs.
        """

        system_instr = get_base_system_prompt(kind)
        user_msg = format_user_prompt(question, aggregated_context)
        response_model = self._get_schema_model(kind)

        try:
            llm_response = self._call_llm(
                system_instr, user_msg, response_model
            )
            LLMClient._log_llm_response(question, kind, llm_response)

        except Exception as e:
            logger.error(f"API Error (answering question): {e}")
            return self._get_fallback_response(kind, str(e))

        # Comparative question schema doesn't have `relevant_pages`
        relevant_pages = getattr(llm_response, "relevant_pages", [])

        return ClientResponse(
            value=llm_response.final_answer,
            relevant_pages=relevant_pages,
            step_by_step_analysis=llm_response.step_by_step_analysis,
            reasoning_summary=llm_response.reasoning_summary,
        )

    def rephrase_comparative_question(
        self, question: str, companies: list[str]
    ) -> RephrasedQuestions:
        """
        Decompose a comparative question into individual questions per company.

        Parameters
        ----------
        question : str
            The original comparative question (e.g., "Who has higher revenue?").
        companies : list[str]
            List of company names involved in the comparison.

        Returns
        -------
        RephrasedQuestions
            A schema containing a list of (company, specific_question) pairs.
            Returns a fallback object (original question for all companies)
            on error.
        """

        system_instr = get_rephrasing_system_prompt()
        user_msg = format_rephrasing_prompt(question, companies)
        response_model = RephrasedQuestions

        try:
            llm_response = self._call_llm(
                system_instr, user_msg, response_model
            )
            LLMClient._log_llm_response(
                question, QuestionKind.COMPARATIVE, llm_response
            )

            return llm_response

        except Exception as e:
            logger.error(f"API Error (rephrasing): {e}")
            logger.warning("Falling back to original question.")

            return RephrasedQuestions(
                questions=[
                    RephrasedQuestion(company_name=c, question=question)
                    for c in companies
                ]
            )
