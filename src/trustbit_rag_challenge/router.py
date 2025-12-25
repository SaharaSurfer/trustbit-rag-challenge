import json

from typing import List, Dict, Any
from loguru import logger

from trustbit_rag_challenge.retriever import ChromaRetriever
from trustbit_rag_challenge.llm.client import LLMClient


class RAGRouter:
    def __init__(self, retriever: ChromaRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm = llm_client

        self.known_companies = self.retriever.company_map.keys()

    def extract_companies(self, text: str) -> List[str]:
        found_companies = []
        for company in self.known_companies:
            if company in text:
                found_companies.append(company.strip('"'))
        
        return found_companies

    def _log_llm_result(self, tag: str, question: str, result: Dict[str, Any]):
        log_payload = {
            "tag": tag,
            "q": question,
            "val": result.get("value"),
            "reason": result.get("reasoning_summary"),
            "full_steps": result.get("step_by_step_analysis")
        }
        logger.info(f"🤖 LLM [{tag}]: {json.dumps(log_payload, ensure_ascii=False)}")

    def answer_question(self, question_text: str, question_kind: str) -> Dict[str, Any]:
        companies = self.extract_companies(question_text)
        
        if not companies:
            logger.warning(f"No known companies found in question: '{question_text}'")
            return self._format_na_response(
                "Company not identified in question", question_kind
            )

        if len(companies) == 1:
            company = companies[0]
            logger.info(f"Routing to SINGLE pipeline: {company}")
            return self._handle_single(company, question_text, question_kind)
        
        else:
            logger.info(f"Routing to COMPARATIVE pipeline: {companies}")
            return self._handle_comparative(companies, question_text)

    def _handle_single(self, company: str, query: str, kind: str) -> Dict[str, Any]:
        top_k = 5
        fetch_k = 30

        question = query.split("?")[0]
        chunks = self.retriever.retrieve(
            question, company_name=company, top_k=top_k, fetch_k=fetch_k)
        
        if not chunks:
            logger.warning(f"No chunks found for {company}")
            return self._format_na_response(
                f"No information found for {company}", kind
            )

        answer = self.llm.generate_answer(query, chunks, kind)
        self._log_llm_result(f"SINGLE:{company}", query, answer)

        return answer

    def _handle_comparative(self, companies: List[str], query: str) -> Dict[str, Any]:
        rephrased_map = self.llm.rephrase_question(query, companies)
        
        individual_results = []
        all_references = []

        for company in companies:
            try:
                sub_query = rephrased_map.get(company, query)
                chunks = self.retriever.retrieve(
                    sub_query, 
                    company_name=company
                )
                
                ans = self.llm.generate_answer(sub_query, chunks, kind="number")
                self._log_llm_result(f"SUB:{company}", sub_query, ans)
                
                val = ans.get("value", "N/A")
                reason = ans.get("reasoning_summary", "")
                
                summary_line = f"Company: {company}\nExtracted Data: {val}\nContext: {reason}\n"
                individual_results.append(summary_line)
                
                all_references.extend(ans['references'])

            except Exception as exc:
                logger.error(f"Error processing {company}: {exc}")
                continue

        aggregated_context = "\n---\n".join(individual_results)
        final_answer = self.llm.generate_comparison(query, aggregated_context)
        self._log_llm_result("COMPARE_FINAL", query, final_answer)
        
        return {
            "value": final_answer["value"],
            "references": all_references,
        }

    def _format_na_response(self, reasoning: str, kind: str) -> Dict[str, Any]:
        if kind == "boolean":
            value = False
        elif kind == "names":
            value = []
        else:
            value = "N/A"

        return {
            "value": value,
            "references": [],
            "step_by_step_analysis": "",
            "reasoning_summary": reasoning 
        }