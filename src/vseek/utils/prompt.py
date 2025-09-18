import logging
import re
from pathlib import Path
from string import Template
from typing import Any


def load_prompt_template(prompt_path: str | Path) -> Template:
    """Load a prompt template from file and return the Template object.

    Args:
        prompt_path: Path to the prompt template file

    Returns:
        Template object that can be formatted later

    Raises:
        ValueError: If the template file cannot be loaded
    """
    if isinstance(prompt_path, str):
        prompt_path = Path(prompt_path)
    try:
        return Template(prompt_path.read_text())
    except Exception as e:
        msg = f"Failed to load prompt from {prompt_path}: {e}"
        logging.error(msg)
        raise ValueError(msg)


def format_prompt_template(
    template: Template, required_variables: bool = True, **kwargs: Any
) -> str:
    """Format a loaded template with provided variables.

    Args:
        template: The Template object to format
        required_variables: Whether to validate that all variables are provided
        **kwargs: Variables to substitute in the template

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If required variables are missing
    """
    variables = re.findall(r"\${(.*?)}", template.template)

    if required_variables:
        for variable in variables:
            if variable not in kwargs:
                msg = f"Variable {variable} not found in kwargs"
                raise ValueError(msg)

    return template.safe_substitute(kwargs)


def load_prompt_as_template(
    prompt_path: str | Path, required_variables: bool = True, **kwargs: Any
) -> str:
    """Load a prompt template and format it with variables (legacy function).

    Args:
        prompt_path: Path to the prompt template file
        required_variables: Whether to validate that all variables are provided
        **kwargs: Variables to substitute in the template

    Returns:
        Formatted prompt string
    """
    template = load_prompt_template(prompt_path)
    return format_prompt_template(template, required_variables, **kwargs)
