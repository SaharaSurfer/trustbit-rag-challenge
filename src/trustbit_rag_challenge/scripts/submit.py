import requests
import pandas as pd

from io import StringIO
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Confirm
from rich.theme import Theme

from trustbit_rag_challenge.config import (
    DATA_DIR,
    SURNAME,
    SUBMISSION_URL, 
    LEADERBOARD_URL
)

custom_theme = Theme({
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold white",
    "primary": "blue",
    "prompt.choices": "blue"
})
console = Console(theme=custom_theme)


def get_latest_submission_file() -> Path:
    """
    Return the most recently modified submission file for the current user.

    The function searches the directory specified by ``DATA_DIR`` for JSON files
    matching the pattern ``submission_<SURNAME>_*.json`` and returns the file
    with the latest modification timestamp.

    Returns
    -------
    pathlib.Path
        Path to the most recently modified submission file.

    Raises
    ------
    FileNotFoundError
        If ``DATA_DIR`` does not exist.
    FileNotFoundError
        If no matching submission files are found in ``DATA_DIR``.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Directory {DATA_DIR} does not exist.")

    files = list(DATA_DIR.glob(f"submission_{SURNAME}_*.json"))
    if not files:
        raise FileNotFoundError(f"No submission files found in {DATA_DIR}!")

    return max(files, key=lambda f: f.stat().st_mtime)


def upload_submission(file_path: Path) -> bool:
    """
    Upload a submission file to the remote submission endpoint.

    The file specified by ``file_path`` is uploaded via a POST request to
    ``SUBMISSION_URL`` using multipart form data. A status spinner and
    user-facing messages are displayed during the upload process.

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
                console.print(f"[success]Submission accepted![/]")
                return True
            else:
                console.print(
                    f"[error]Submission failed "
                    f"[Code {response.status_code}][/]"
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
        ascending=True
    ).reset_index(drop=True)

    table = Table(
        title="🏆 RAG Challenge Leaderboard",
        title_style="highlight",
        box=box.ROUNDED,
        header_style="primary",
        border_style="primary",
    )

    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
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
    it locates the most recent submission file, asks the user for
    confirmation before uploading it, performs the upload if
    confirmed, and finally fetches and displays the leaderboard.

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
    try:
        latest_file = get_latest_submission_file()
    except FileNotFoundError as e:
        console.print(f"[error]Error:[/] {e}")
        return

    console.print(f"Found latest submission: [highlight]{latest_file.name}[/]")
    
    if Confirm.ask("Do you want to submit this file?", console=console):
        success = upload_submission(latest_file)
        if success:
            console.print("\n")
        else:
            console.print(f"[error]Something went wrong during submission")
    
    fetch_and_display_leaderboard()
