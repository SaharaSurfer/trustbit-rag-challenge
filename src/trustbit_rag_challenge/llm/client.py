import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from trustbit_rag_challenge.config import OPENAI_MODEL

from .prompts import (
    format_rephrasing_prompt,
    format_user_prompt,
    get_base_system_prompt,
    get_rephrasing_system_prompt,
)
from .schemas import (
    BaseModel,
    BooleanResponse,
    ComparativeResponse,
    NameResponse,
    NamesResponse,
    NumberResponse,
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

    def _get_schema_model(self, kind: str) -> type[BaseModel]:
        mapping: dict[str, type[BaseModel]] = {
            "number": NumberResponse,
            "boolean": BooleanResponse,
            "name": NameResponse,
            "names": NamesResponse,
            "comparative": ComparativeResponse,
        }
        return mapping.get(kind, NameResponse)

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def generate_answer(
        self, question: str, chunks: list[dict], kind: str
    ) -> dict[str, Any]:
        context_str = "\n\n".join(
            [
                f"--- Page {c.get('page_index', '?')} ---\n{c['text']}"
                for c in chunks
            ]
        )

        system_instr = get_base_system_prompt(kind)
        user_msg = format_user_prompt(question, context_str)
        response_model = self._get_schema_model(kind)

        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                text_format=response_model,
            )

            content = response.output_parsed
            value = content.final_answer
            references = self._filter_references(chunks, content.relevant_pages)

            if str(value).upper() == "N/A":
                references = []

            if kind == "boolean" and value is False:
                references = []

            if kind == "names" and isinstance(value, list) and not value:
                references = []

            return {
                "value": value,
                "references": references,
                "step_by_step_analysis": content.step_by_step_analysis,
                "reasoning_summary": content.reasoning_summary,
            }

        except Exception as e:
            logger.error(f"API Error (generate_answer): {e}")
            return self._get_fallback_response(kind, str(e))

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def generate_comparison(
        self, question: str, aggregated_context: str
    ) -> dict[str, Any]:
        system_instr = get_base_system_prompt("comparative")
        user_msg = format_user_prompt(question, aggregated_context)
        response_model = ComparativeResponse

        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                text_format=response_model,
            )

            content = response.output_parsed

            return {
                "value": content.final_answer,
                "references": [],
                "step_by_step_analysis": content.step_by_step_analysis,
                "reasoning_summary": content.reasoning_summary,
            }

        except Exception as e:
            logger.error(f"API Error (generate_comparison): {e}")
            return {
                "value": "N/A",
                "references": [],
                "step_by_step_analysis": "",
                "reasoning_summary": f"Error: {str(e)}",
            }

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def rephrase_question(
        self, question: str, companies: list[str]
    ) -> dict[str, str]:
        system_instr = get_rephrasing_system_prompt()
        user_msg = format_rephrasing_prompt(question, companies)

        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                text_format=RephrasedQuestions,
            )

            content = response.output_parsed

            result_dict = {}
            for item in content.questions:
                clean_name = item.company_name.strip()
                result_dict[clean_name] = item.question

            return result_dict

        except Exception as e:
            logger.error(f"Rephrasing Error: {e}")
            logger.warning("Falling back to original question.")
            return {company: question for company in companies}

    def _filter_references(
        self, chunks: list[dict], mentioned_pages: list[int]
    ) -> list[dict]:
        if not mentioned_pages:
            return []

        filtered = []
        mentioned_set = set(mentioned_pages)
        for c in chunks:
            p_idx = int(c.get("page_index", -1))
            if p_idx in mentioned_set:
                filtered.append({"pdf_sha1": c["source"], "page_index": p_idx})

        return filtered

    def _get_fallback_response(self, kind: str, error_msg: str) -> dict:
        val: str | bool | list[str] = "N/A"
        if kind == "boolean":
            val = False
        elif kind == "names":
            val = []

        return {
            "value": val,
            "references": [],
            "step_by_step_analysis": "",
            "reasoning_summary": f"Error: {error_msg}",
        }
