# Local Assistant
# Local Assistant

This small tool is part of the `sarah` toolkit. It provides a local voice-enabled assistant. It is intentionally small and safe — it will only execute shell commands after you explicitly ask to run them and confirm.

## Quick start

1. Install dependencies (preferably in a virtualenv):

```powershell
python -m pip install -r tools/assistant/requirements.txt
```

2. Run the assistant:

```powershell
python tools/assistant/assistant.py
```

## Usage notes

- Speak/text replies are produced using `pyttsx3` (offline TTS).
- If you set `OPENAI_API_KEY` in your environment and have the `openai` package installed, replies will come from OpenAI's completion API; otherwise the assistant echoes your prompt.
- To execute a command, type `run: <your command>` (or `execute: <your command>`). The assistant will ask for confirmation before running the command.

## Security

Only use this tool on machines you trust. Do not run untrusted commands. The assistant requires your explicit confirmation before running anything.

## Example

You: run: dir
Assistant: You asked to run: dir. Do you confirm? Type 'yes' to confirm.
Confirm (yes/no): yes
Assistant runs the command and speaks a short summary of the output.

## New features

- **Preview / dry-run for Excel steps**: when you apply a step (e.g. `step "Total = Price * Quantity" file out.xlsx sheet Sheet1`), the assistant can show a preview of the top rows after applying the step before writing the file. Use the `Preview before saving` or `Preview (yes/no)` prompts in the Excel mini-REPL.
- **Safer parsing**: paths and expressions with spaces should be quoted (the mini-REPL uses `shlex.split`). Example:
	- `load csv "data/my file.csv" into "out file.xlsx" sheet "Sheet 1"`
- **Image OCR**: use `image` mode and `read image <path>` to extract text using Tesseract (needs Tesseract binary installed).

## Examples

- Start the assistant and enter Excel mode:
	- `excel` then: `load csv "tools/assistant/example.csv" into "out.xlsx" sheet "Sheet1"`
	- `step "Total = Price * Quantity" file out.xlsx sheet Sheet1` (choose preview then confirm)

Security reminder: the assistant will always ask for confirmation before writing files or executing shell commands.

---

If you want more features (preview UI, advanced Excel transformations, or integration with real OpenAI chat models), tell me which one and I will add it.
