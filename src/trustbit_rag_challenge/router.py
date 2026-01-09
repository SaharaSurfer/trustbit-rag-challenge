from loguru import logger

from trustbit_rag_challenge.enums import QuestionKind
from trustbit_rag_challenge.llm.client import LLMClient
from trustbit_rag_challenge.retriever import ChromaRetriever
from trustbit_rag_challenge.schemas import ClientResponse, RouterResponse


class RAGRouter:
    """
    Orchestration logic for the RAG pipeline.

    This class routes incoming questions to the appropriate processing pipeline
    (Single Company vs. Comparative) and manages the flow of data between the
    Retriever and the LLMClient

    Attributes
    ----------
    retriever : ChromaRetriever
        Instance of the retrieval system for fetching document chunks.
    llm : LLMClient
        Instance of the LLM client for generating answers.
    known_companies : list[str]
        A cached list of company names available in the knowledge base, used
        for entity extraction from questions.
    """

    def __init__(self, retriever: ChromaRetriever, llm_client: LLMClient):
        """
        Initialize the router.

        Parameters
        ----------
        retriever : ChromaRetriever
            The initialized retriever instance.
        llm_client : LLMClient
            The initialized LLM client instance.
        """

        self.retriever = retriever
        self.llm = llm_client

        self.known_companies = self.retriever.company_map.keys()

    @staticmethod
    def _validate_references(
        kind: QuestionKind, chunks: list[dict], client_response: ClientResponse
    ) -> list[dict]:
        """
        Filter and validate the references provided by the LLM.

        This method maps the page numbers returned by the LLM back to the
        original document metadata (`pdf_sha1`). It also enforces that
        negative answers ("N/A", False, empty lists) must not have references.

        Parameters
        ----------
        kind : QuestionKind
            The type of the question (used to determine negative value logic).
        chunks : list[dict]
            The list of chunks that were provided as context to the LLM.
        client_response : ClientResponse
            The structured response received from the LLM.

        Returns
        -------
        list[dict]
            A list of reference objects containing `pdf_sha1` and `page_index`.
            Returns an empty list if the answer is negative or no pages match.
        """

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
        """
        Identify which known companies are mentioned in the question.

        Parameters
        ----------
        text : str
            The user question text.

        Returns
        -------
        list[str]
            A list of unique company names found in the text.
        """

        found_companies = []
        for company in self.known_companies:
            if company in text:
                found_companies.append(company.strip('"'))

        return found_companies

    def answer_question(
        self, question_text: str, question_kind: QuestionKind
    ) -> RouterResponse:
        """
        Main entry point for answering a question.

        Routes the question based on the number of companies identified:
        - 0 companies: Returns a fallback error response.
        - 1 company: Routes to `_handle_single`.
        - >1 companies: Routes to `_handle_comparative`.

        Parameters
        ----------
        question_text : str
            The raw text of the question.
        question_kind : QuestionKind
            The expected type of the answer (e.g., NUMBER, BOOLEAN).

        Returns
        -------
        RouterResponse
            The final answer object containing the value and references.
        """

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
        """
        Process a question for a single company.

        Pipeline:
        1. Clean the query.
        2. Retrieve relevant chunks.
        3. Generate answer using LLM.
        4. Validate page references.

        Parameters
        ----------
        company : str
            The target company name.
        query : str
            The question text.
        kind : QuestionKind
            The type of answer expected.

        Returns
        -------
        tuple[ClientResponse, list]
            A tuple containing the raw LLM response DTO and the list of
            validated reference dictionaries.
        """

        top_k = 5
        fetch_k = 30

        # Heuristic to remove trailing noise found in questions
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
        """
        Process a comparative question involving multiple companies.

        Pipeline:
        1. Decompose the question using LLM to rephrase the query.
        2. Iterate through companies and call `_handle_single` for each.
        3. Combine extracted data into a summary context.
        4. Generate final comparison using LLM.

        Parameters
        ----------
        companies : list[str]
            List of companies involved in the comparison.
        query : str
            The original comparative question.

        Returns
        -------
        tuple[ClientResponse, list]
            A tuple containing the final comparative answer and the aggregated
            list of references from all companies.
        """

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

            summary_line = f"Company: {company}\nExtracted Data: {ans.value}\n"
            individual_results.append(summary_line)
            all_references.extend(references)

        aggregated_context = "\n".join(individual_results)
        final_answer = self.llm.answer_question(
            query, aggregated_context, kind=QuestionKind.COMPARATIVE
        )

        return final_answer, all_references
