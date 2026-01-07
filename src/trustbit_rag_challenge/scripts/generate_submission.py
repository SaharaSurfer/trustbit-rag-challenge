import json

from loguru import logger
from tqdm import tqdm

from trustbit_rag_challenge.config import (
    DATA_DIR,
    QUESTIONS_PATH,
    SUBMISSION_NAME,
    TEAM_EMAIL,
)
from trustbit_rag_challenge.llm.client import LLMClient
from trustbit_rag_challenge.llm.schemas import (
    Answer,
    AnswerSubmission,
    SourceReference,
)
from trustbit_rag_challenge.logging_utils import setup_logging
from trustbit_rag_challenge.retriever import ChromaRetriever
from trustbit_rag_challenge.router import RAGRouter


def process_single_question(q_data: dict, orchestrator: RAGRouter) -> Answer:
    text = q_data["text"]
    kind = q_data["kind"]

    try:
        result = orchestrator.answer_question(text, kind)

        refs = []
        for r in result.get("references", []):
            refs.append(
                SourceReference(
                    pdf_sha1=r["pdf_sha1"], page_index=r["page_index"]
                )
            )

        return Answer(
            question_text=text,
            kind=kind,
            value=result["value"],
            references=refs,
        )

    except Exception as e:
        logger.error(f"Error processing question '{text[:50]}...': {e}")
        val: str | bool | list[str] = "N/A"
        if kind == "boolean":
            val = False
        if kind == "names":
            val = []

        return Answer(question_text=text, kind=kind, value=val, references=[])


def main():
    setup_logging()

    logger.info("Initializing RAG System components...")

    try:
        retriever = ChromaRetriever()
        llm_client = LLMClient()
        orchestrator = RAGRouter(retriever, llm_client)

    except Exception as e:
        logger.critical(f"Failed to initialize RAG system: {e}")
        return

    if not QUESTIONS_PATH.exists():
        logger.error(f"Questions file not found at {QUESTIONS_PATH}")
        return

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions_data = json.load(f)

    logger.info(f"Loaded {len(questions_data)} questions.")
    logger.info("Starting sequential processing...")

    answers_list = []

    for i, q_data in enumerate(tqdm(questions_data, desc="Processing")):
        ans = process_single_question(q_data, orchestrator)
        answers_list.append(ans)

        logger.debug(
            f"Processed {i + 1}/{len(questions_data)}: "
            f"Last answer val='{ans.value}'"
        )

    submission = AnswerSubmission(
        team_email=TEAM_EMAIL,
        submission_name=SUBMISSION_NAME,
        answers=answers_list,
    )

    output_filename = f"submission_{SUBMISSION_NAME}.json"
    output_path = DATA_DIR / output_filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission.model_dump(), f, ensure_ascii=False, indent=2)

        logger.success(f"Submission saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save submission file: {e}")
