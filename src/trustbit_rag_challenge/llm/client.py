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
    def __init__(self):
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
        log_payload = {"q": question, "kind": kind, **result.model_dump()}
        logger.info(f"🤖 LLM: {json.dumps(log_payload, ensure_ascii=False)}")

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def _call_llm(
        self, system_instr: str, user_msg: str, response_model: type[BaseModel]
    ) -> Any:
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
