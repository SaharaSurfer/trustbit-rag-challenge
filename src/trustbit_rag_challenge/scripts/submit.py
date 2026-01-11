from io import StringIO
from pathlib import Path
from typing import Literal

import pandas as pd
import questionary
import requests
from rich import box
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from trustbit_rag_challenge.config import (
    DATA_DIR,
    LEADERBOARD_URL,
    SUBMISSION_URL,
    SURNAME,
)

custom_theme = Theme(
    {
        "error": "bold red",
        "success": "bold green",
        "highlight": "bold white",
        "primary": "#7fbbb3",
    }
)
console = Console(theme=custom_theme)


def select_submission_file() -> Path | None:
    """
    Interactively select a submission file using a CLI menu.

    Scans ``DATA_DIR`` for files matching ``submission_<SURNAME>_*.json``,
    sorts them by modification time (newest first), and presents an
    interactive list navigable with arrow keys.

    Returns
    -------
    pathlib.Path or None
        The path to the selected file, or ``None`` if the user cancels
        or no files are found.

    Raises
    ------
    FileNotFoundError
        If ``DATA_DIR`` does not exist.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Directory {DATA_DIR} does not exist.")

    files = list(DATA_DIR.glob(f"submission_{SURNAME}_*.json"))
    if not files:
        console.print(f"[error]No submission files found in {DATA_DIR}![/]")
        return None

    # Newest first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    choices = []
    for f in files:
        choices.append(questionary.Choice(title=f"{f.name}", value=f))

    choices.append(questionary.Choice(title="cancel"))

    selection = questionary.select(
        message="Select a file to submit:",
        qmark="",
        choices=choices,
        use_indicator=True,
        style=questionary.Style(
            [
                ("question", "bold"),
                ("answer", "fg:#7fbbb3 bold"),
                ("pointer", "fg:#7fbbb3 bold"),
                ("highlighted", "fg:#7fbbb3 bold"),
            ]
        ),
    ).ask()

    return selection


def upload_submission(file_path: Path) -> bool:
    """
    Upload a submission file to the remote submission endpoint.

    The file specified by ``file_path`` is uploaded via a POST request to
    ``SUBMISSION_URL``. A status spinner and user-facing messages are
    displayed during the upload process.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the submission file to be uploaded.

    Returns
    -------
    bool
        ``True`` if the submission was successfully accepted by the server
        (HTTP 200 response), ``False`` otherwise.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist or cannot be opened.
    """
    filename = file_path.name

    with console.status(
        f"[primary]Uploading[/] [highlight]{filename}[/]...", spinner="arc"
    ):
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    SUBMISSION_URL, files=files, timeout=30
                )

            if response.status_code == 200:
                console.print("[success]Submission accepted![/]")
                return True
            else:
                console.print(
                    f"[error]Submission failed [Code {response.status_code}][/]"
                )
                console.print(f"[dim]{response.text}[/]")
                return False

        except requests.RequestException as e:
            console.print(f"[error]Network error:[/]\n{e}")
            return False


def fetch_and_display_leaderboard() -> None:
    """
    Fetch the leaderboard data from the remote endpoint and display it
    as a formatted table in the console.

    The function retrieves CSV-formatted leaderboard data from
    ``LEADERBOARD_URL``, parses it into a pandas DataFrame, sorts the
    entries by the ``rank`` column, and renders the top 15 rows using
    Rich's ``Table`` component. Rows containing the user's surname are
    visually highlighted.

    This function is intended for interactive use and produces console
    output as a side effect. It does not return any value.

    Returns
    -------
    None

    Notes
    -----
    - The leaderboard is sorted numerically by the ``rank`` column
      (non-numeric values are coerced to ``NaN``).
    - If the leaderboard is empty or unavailable, a user-friendly
      message is printed and the function exits early.
    - Rows containing ``SURNAME`` in any column are highlighted.

    Raises
    ------
    None
        All exceptions raised during network access or CSV parsing are
        caught and reported to the console.
    """
    with console.status("[primary]Fetching leaderboard...[/]", spinner="arc"):
        try:
            response = requests.get(LEADERBOARD_URL, timeout=10)
            if response.status_code != 200:
                console.print(
                    f"[error]Failed to fetch leaderboard "
                    f"[Code {response.status_code}][/]"
                )
                return

            csv_data = response.text.strip()
            if not csv_data:
                console.print("[dim]Leaderboard is currently empty.[/]")
                return

            df = pd.read_csv(StringIO(csv_data))

        except Exception as e:
            console.print(f"[error]Error processing leaderboard data:[/]\n{e}")
            return

    if df.empty:
        console.print("[dim]Leaderboard table is empty.[/]")
        return

    df = df.sort_values(
        by="rank",
        key=lambda col: pd.to_numeric(col, errors="coerce"),
        ascending=True,
    ).reset_index(drop=True)

    table = Table(
        title="🏆 RAG Challenge Leaderboard",
        title_style="highlight",
        box=box.ROUNDED,
        header_style="primary",
        border_style="primary",
    )

    for col in df.columns:
        justify: Literal["right", "left"] = (
            "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        )
        table.add_column(str(col).upper(), justify=justify)

    for _, row in df.head(15).iterrows():
        row_style = ""

        is_my_submission = False
        for val in row.values:
            val_str = str(val)
            if SURNAME in val_str:
                is_my_submission = True
                break

        if is_my_submission:
            row_style = "highlight"

        cells = [str(item) if not pd.isna(item) else "-" for item in row]
        table.add_row(*cells, style=row_style)

    console.print(table)


def main() -> None:
    """
    Entry point for the submission workflow.

    This function orchestrates the end-to-end user interaction:
    it asks the user to choose the submission file before uploading it,
    performs the upload, and finally fetches and displays the leaderboard.

    All user interaction and output are handled via the console.
    Errors related to missing submission files are caught and
    reported in a user-friendly way.

    Returns
    -------
    None

    Notes
    -----
    - This function is intended to be used as the main CLI entry
      point (e.g., invoked via a Poetry script).
    - Network or upload failures are handled by the called
      functions and do not raise exceptions here.

    Raises
    ------
    None
        All expected exceptions are handled internally.
    """

    selected_file = select_submission_file()
    if isinstance(selected_file, Path):
        success = upload_submission(selected_file)
        if not success:
            console.print("[error]Something went wrong during submission")

    fetch_and_display_leaderboard()
