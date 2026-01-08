from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from trustbit_rag_challenge.enums import QuestionKind

CURRENT_DIR = Path(__file__).parent
TEMPLATES_DIR = CURRENT_DIR / "templates"

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **kwargs) -> str:
    """
    Render a specific Jinja2 template with provided variables.

    Parameters
    ----------
    template_name : str
        The filename of the template to render (e.g., 'system_base.j2').
        Must exist in the `templates` directory.
    **kwargs
        Arbitrary keyword arguments to pass to the template engine.

    Returns
    -------
    str
        The rendered string.

    Raises
    ------
    jinja2.TemplateNotFound
        If the template file does not exist.
    """

    template = env.get_template(template_name)
    return template.render(**kwargs)


def _render_optional_template(template_name: str) -> str:
    """
    Attempt to render a template, returning an empty string if missing.

    Helper function for loading optional prompt components (like specific
    instructions or examples for a given question type).

    Parameters
    ----------
    template_name : str
        The filename of the template.

    Returns
    -------
    str
        The rendered template content, or an empty string if the file is found.
    """

    try:
        return render_template(template_name)
    except TemplateNotFound:
        return ""


def get_base_system_prompt(kind: QuestionKind) -> str:
    """
    Construct the full system prompt for RAG extraction tasks.

    Assembles the prompt by:
    1. Loading the base system instruction (`system_base.j2`).
    2. Injecting specific instructions for the `kind` if they exist
       (`instruction_{kind}.j2`).
    3. Injecting examples for the `kind` if they exist
       (`examples_{kind}.j2`).

    Parameters
    ----------
    kind : QuestionKind
        The type of question (NUMBER, BOOLEAN, etc.) which dictates
        the specific instructions and examples to load.

    Returns
    -------
    str
        The fully assembled system prompt.
    """

    instruction = _render_optional_template(f"instruction_{kind}.j2")
    examples = _render_optional_template(f"examples_{kind}.j2")

    return render_template(
        "system_base.j2", specific_instruction=instruction, examples=examples
    )


def format_user_prompt(question: str, context_str: str) -> str:
    """
    Format the user message combining the question and retrieved context.

    Parameters
    ----------
    question : str
        The user's question.
    context_str : str
        The aggregated text from retrieved document chunks (usually formatted
        with page delimiters).

    Returns
    -------
    str
        The formatted user prompt ready for the LLM.
    """

    return render_template(
        "query_user.j2", context_str=context_str, question=question
    )


def get_rephrasing_system_prompt() -> str:
    """
    Load the system prompt for the query decomposition (rephrasing) task.

    This prompt instructs the LLM to break down comparative questions
    into individual sub-questions.

    Returns
    -------
    str
        The system prompt for rephrasing.
    """

    examples = render_template("examples_rephrasing.j2")

    return render_template("system_rephrasing.j2", examples=examples)


def format_rephrasing_prompt(question: str, companies: list[str]) -> str:
    """
    Format the user message for the query decomposition task.

    Parameters
    ----------
    question : str
        The original comparative question (e.g., "Who has higher revenue...").
    companies : list[str]
        A list of company names identified in the question.

    Returns
    -------
    str
        The formatted prompt containing the question and the target companies.
    """

    clean_companies = [c.strip('"').strip("'") for c in companies]

    return render_template(
        "query_rephrasing.j2", question=question, companies=clean_companies
    )
