import json

from loguru import logger
from tqdm import tqdm

from trustbit_rag_challenge.config import (
    DATA_DIR,
    QUESTIONS_PATH,
    SUBMISSION_NAME,
    TEAM_EMAIL,
)
from trustbit_rag_challenge.enums import QuestionKind
from trustbit_rag_challenge.llm.client import LLMClient
from trustbit_rag_challenge.logging_utils import setup_logging
from trustbit_rag_challenge.retriever import ChromaRetriever
from trustbit_rag_challenge.router import RAGRouter
from trustbit_rag_challenge.schemas import (
    Answer,
    AnswerSubmission,
    SourceReference,
)


def process_single_question(q_data: dict, orchestrator: RAGRouter) -> Answer:
    """
    Process a single question dictionary and generate a formatted Answer object.

    This function handles:
    1. Parsing the question type into a `QuestionKind` enum.
    2. Invoking the RAG orchestration logic.
    3. Mapping reference dictionaries to `SourceReference` Pydantic models.
    4. Providing a fail-safe fallback mechanism in case of runtime errors.

    Parameters
    ----------
    q_data : dict
        A dictionary containing the raw question data (keys: 'text', 'kind').
    orchestrator : RAGRouter
        The initialized RAG orchestration logic handler.

    Returns
    -------
    Answer
        A validated Pydantic object ready for inclusion in the submission file.
        If an error occurs, returns a valid 'N/A' or empty Answer object to
        prevent the pipeline from crashing.
    """

    text = q_data["text"]
    kind = QuestionKind(q_data["kind"])

    try:
        result = orchestrator.answer_question(text, kind)

        refs = []
        for r in result.references:
            refs.append(
                SourceReference(
                    pdf_sha1=r["pdf_sha1"], page_index=r["page_index"]
                )
            )

        return Answer(
            question_text=text,
            kind=kind,
            value=result.value,
            references=refs,
        )

    except Exception as e:
        logger.error(f"Error processing question '{text[:50]}...': {e}")
        val: str | bool | list[str] = "N/A"
        if kind == QuestionKind.BOOLEAN:
            val = False
        if kind == QuestionKind.NAMES:
            val = []

        return Answer(question_text=text, kind=kind, value=val, references=[])


def main():
    """
    Execute the full RAG pipeline to generate the final submission file.

    Workflow:
    1.  **Setup**: Configures logging and initializes heavy components.
    2.  **Load**: Reads the input questions file specified in the config.
    3.  **Process**: Iterates sequentially through all questions, generating
        answers via the `RAGRouter`. Sequential processing is used to ensure
        stability and avoid rate limits.
    4.  **Save**: Serializes the results into a JSON file strictly following
        the challenge's submission schema.

    Side Effects
    ------------
    - Writes a JSON submission file to `data/submission_<name>.json`.
    """

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
