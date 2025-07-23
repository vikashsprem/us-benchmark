"""
For each format, this code extracts the largest HTML table from the response.
"""

import json
from typing import Any
import os
import re


def parse_unsiloedvl_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    # Extract HTML table from the response
    # First look for a complete table tag
    html_content = data.get("html_table", "")

    # If not found directly, try to find table in the raw text
    if not html_content:
        raw_response = data.get("raw_response", "")

        # Try to extract an HTML table
        table_pattern = r"<table>.*?</table>"
        table_match = re.search(table_pattern, raw_response, re.DOTALL)

        if table_match:
            html_content = table_match.group(0)

    # If still no table content, try markdown tables
    if not html_content or "<table>" not in html_content:
        raw_response = data.get("raw_response", "")

        # Convert markdown table to HTML if present
        if "|" in raw_response and "-|-" in raw_response:
            # Very simple markdown to HTML conversion
            lines = [
                line.strip()
                for line in raw_response.split("\n")
                if line.strip() and "|" in line
            ]
            if len(lines) > 1:
                html_content = "<table>"
                for i, line in enumerate(lines):
                    # Skip separator line (contains only | and -)
                    if i == 1 and all(c in "|-:" for c in line):
                        continue

                    cells = [cell.strip() for cell in line.split("|")]
                    cells = [cell for cell in cells if cell]  # Remove empty cells

                    if cells:
                        tag = "th" if i == 0 else "td"
                        html_content += (
                            "<tr>"
                            + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
                            + "</tr>"
                        )

                html_content += "</table>"

    return html_content if html_content else None, data


def parse_llama_response(path: str) -> tuple[str | None, Any]:
    """
    Parse LLaMA model responses for table extraction.
    Handles the JSON format created by llama_inference.py
    """
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    # Check if this is an error result
    if data.get("status") == "error":
        return None, data

    # Extract HTML table from the response
    html_content = data.get("html_table", "")

    # If not found directly, try to find table in the model response
    if not html_content:
        model_response = data.get("model_response", "")

        # Try to extract an HTML table
        table_pattern = r"<table>.*?</table>"
        table_match = re.search(table_pattern, model_response, re.DOTALL)

        if table_match:
            html_content = table_match.group(0)

    # If still no table content, try the full generated text
    if not html_content or "<table>" not in html_content:
        generated_text = data.get("generated_text", "")

        # Try to extract an HTML table from generated text
        table_pattern = r"<table>.*?</table>"
        table_match = re.search(table_pattern, generated_text, re.DOTALL)

        if table_match:
            html_content = table_match.group(0)

    # If still no HTML table, try to convert markdown tables
    if not html_content or "<table>" not in html_content:
        text_to_search = data.get("model_response", "") or data.get("generated_text", "")

        # Convert markdown table to HTML if present
        if "|" in text_to_search and "-|-" in text_to_search:
            lines = [
                line.strip()
                for line in text_to_search.split("\n")
                if line.strip() and "|" in line
            ]
            if len(lines) > 1:
                html_content = "<table>"
                for i, line in enumerate(lines):
                    # Skip separator line (contains only | and -)
                    if i == 1 and all(c in "|-:" for c in line):
                        continue

                    cells = [cell.strip() for cell in line.split("|")]
                    cells = [cell for cell in cells if cell]  # Remove empty cells

                    if cells:
                        tag = "th" if i == 0 else "td"
                        html_content += (
                            "<tr>"
                            + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
                            + "</tr>"
                        )

                html_content += "</table>"

    return html_content if html_content else None, data
