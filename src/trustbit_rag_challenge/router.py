from loguru import logger

from trustbit_rag_challenge.enums import QuestionKind
from trustbit_rag_challenge.llm.client import LLMClient
from trustbit_rag_challenge.llm.schemas import ClientResponse, RouterResponse
from trustbit_rag_challenge.retriever import ChromaRetriever


class RAGRouter:
    def __init__(self, retriever: ChromaRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm = llm_client

        self.known_companies = self.retriever.company_map.keys()

    @staticmethod
    def _validate_references(
        kind: QuestionKind, chunks: list[dict], client_response: ClientResponse
    ) -> list[dict]:
        relevant_pages = client_response.relevant_pages
        if not relevant_pages:
            return []

        # Clear references if llm answer is negative
        value = client_response.value
        if (
            str(value).upper() == "N/A"
            or (kind == QuestionKind.BOOLEAN and value is False)
            or (kind == QuestionKind.NAMES and value == [])
        ):
            return []

        filtered = []
        mentioned_set = set(relevant_pages)
        for c in chunks:
            p_idx = int(c.get("page_index", -1))
            if p_idx in mentioned_set:
                filtered.append({"pdf_sha1": c["source"], "page_index": p_idx})

        return filtered

    def _extract_companies(self, text: str) -> list[str]:
        found_companies = []
        for company in self.known_companies:
            if company in text:
                found_companies.append(company.strip('"'))

        return found_companies

    def answer_question(
        self, question_text: str, question_kind: QuestionKind
    ) -> RouterResponse:
        companies = self._extract_companies(question_text)

        if len(companies) == 1:
            company = companies[0]
            logger.info(f"Routing to SINGLE pipeline: {company}")
            ans, refs = self._handle_single(
                company, question_text, question_kind
            )

        elif len(companies) > 1:
            logger.info(f"Routing to COMPARATIVE pipeline: {companies}")
            ans, refs = self._handle_comparative(companies, question_text)

        else:
            logger.warning(
                f"No known companies found in question: '{question_text}'"
            )
            ans, refs = (
                LLMClient._get_fallback_response(
                    question_kind, "Company not identified in question"
                ),
                [],
            )

        return RouterResponse(value=ans.value, references=refs)

    def _handle_single(
        self, company: str, query: str, kind: QuestionKind
    ) -> tuple[ClientResponse, list]:
        top_k = 5
        fetch_k = 30

        question = query.split("?")[0]
        chunks = self.retriever.retrieve(
            question, company_name=company, top_k=top_k, fetch_k=fetch_k
        )

        if not chunks:
            logger.warning(f"No chunks found for {company}")
            fallback_response = LLMClient._get_fallback_response(
                kind, f"No information found for {company}"
            )
            return fallback_response, []

        aggregated_context = "\n\n".join(
            [
                f"--- Page {c.get('page_index', '?')} ---\n{c['text']}"
                for c in chunks
            ]
        )
        answer = self.llm.answer_question(query, aggregated_context, kind)
        references = RAGRouter._validate_references(kind, chunks, answer)

        return answer, references

    def _handle_comparative(
        self, companies: list[str], query: str
    ) -> tuple[ClientResponse, list]:
        rephrased_questions = self.llm.rephrase_comparative_question(
            query, companies
        )
        rephrased_map = {
            item.company_name.strip(): item.question
            for item in rephrased_questions.questions
        }

        individual_results = []
        all_references = []

        for company in companies:
            sub_query = rephrased_map.get(company, query)
            ans, references = self._handle_single(
                company, sub_query, QuestionKind.NUMBER
            )

            summary_line = (
                f"Company: {company}\n"
                f"Extracted Data: {ans.value}\n"
                f"Context: {ans.reasoning_summary}\n"
            )
            individual_results.append(summary_line)
            all_references.extend(references)

        aggregated_context = "\n---\n".join(individual_results)
        final_answer = self.llm.answer_question(
            query, aggregated_context, kind=QuestionKind.COMPARATIVE
        )

        return final_answer, all_references
