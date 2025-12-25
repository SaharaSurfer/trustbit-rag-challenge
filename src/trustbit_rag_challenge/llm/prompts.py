from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

CURRENT_DIR = Path(__file__).parent
TEMPLATES_DIR = CURRENT_DIR / "templates"

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(
        enabled_extensions=('html', 'xml'), default_for_string=False
    ),
    trim_blocks=True,
    lstrip_blocks=True
)


def render_template(template_name: str, **kwargs) -> str:
    template = env.get_template(template_name)
    return template.render(**kwargs)


def get_base_system_prompt(kind: str) -> str:
    instruction = ""
    if kind == "comparative":
        instruction = render_template("instruction_comparative.j2")

    examples = ""
    if kind in ["number", "boolean", "name", "names", "comparative"]:
        examples = render_template(f"examples_{kind}.j2")
    
    return render_template(
        "system_base.j2",
        specific_instruction=instruction,
        examples=examples
    )


def format_user_prompt(question: str, context_str: str) -> str:
    return render_template(
        "query_user.j2", context_str=context_str, question=question
    )


def get_rephrasing_system_prompt() -> str:
    examples = render_template("examples_rephrasing.j2")

    return render_template(
        "system_rephrasing.j2",
        examples=examples
    )


def format_rephrasing_prompt(question: str, companies: list[str]) -> str:
    quote = '"'
    companies_str = ", ".join([f'"{c.strip(quote)}"' for c in companies])
    
    return render_template(
        "query_rephrasing.j2",
        question=question,
        companies_str=companies_str
    )