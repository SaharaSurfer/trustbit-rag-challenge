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
    template = env.get_template(template_name)
    return template.render(**kwargs)


def _render_optional_template(template_name: str) -> str:
    try:
        return render_template(template_name)
    except TemplateNotFound:
        return ""


def get_base_system_prompt(kind: QuestionKind) -> str:
    instruction = _render_optional_template(f"instruction_{kind}.j2")
    examples = _render_optional_template(f"examples_{kind}.j2")

    return render_template(
        "system_base.j2", specific_instruction=instruction, examples=examples
    )


def format_user_prompt(question: str, context_str: str) -> str:
    return render_template(
        "query_user.j2", context_str=context_str, question=question
    )


def get_rephrasing_system_prompt() -> str:
    examples = render_template("examples_rephrasing.j2")

    return render_template("system_rephrasing.j2", examples=examples)


def format_rephrasing_prompt(question: str, companies: list[str]) -> str:
    clean_companies = [c.strip('"').strip("'") for c in companies]

    return render_template(
        "query_rephrasing.j2", question=question, companies=clean_companies
    )
