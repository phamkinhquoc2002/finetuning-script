from dataclasses import dataclass
from typing import Literal, Union
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

@dataclass
class LogMessage(TypedDict):
    """LogMessage class."""
    type: Literal['ERROR', 'OUTPUT_MESSAGE', 'INFO']
    text: str

def log_message(log: LogMessage) -> None:
    """
    Log message beautifully.

    Parameters:
        type (LogMessage): Type of Logging.
        text: Logging Message
    """
    console = Console()
    formatted_text = Text()

    formatted_text.append(log['text'], style="white")
    if log["type"]== 'ERROR':
        border_style = 'red'
    elif log["type"] == 'OUTPUT_MESSAGE':
        border_style = 'green'
    elif log["type"] == 'INFO':
        border_style = 'yellow'
    console.print(Panel(formatted_text, title=log["type"], 
                        title_align="center", 
                        border_style=border_style))