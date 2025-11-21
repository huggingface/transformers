#!/usr/bin/env python3
"""
Local voice-enabled assistant.

Features:
- Speaks replies using `pyttsx3` (offline TTS).
- Optionally uses OpenAI if `OPENAI_API_KEY` is set in the environment.
- Allows executing shell commands only after interactive confirmation.

Usage:
    python assistant.py

Security:
This tool runs commands on your machine only after you type a run command and confirm.
Be careful and only run in a trusted environment.
"""
import os
import subprocess
import threading
import sys
from pathlib import Path

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None
    _OPENAI_AVAILABLE = False

# Ensure the repository and the assistant package directory are on sys.path so
# imports work whether running as a script or as a package.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR
for _ in range(4):
    if (_REPO_ROOT / 'pyproject.toml').exists() or (_REPO_ROOT / '.git').exists():
        break
    if _REPO_ROOT.parent == _REPO_ROOT:
        break
    _REPO_ROOT = _REPO_ROOT.parent
# Insert assistant directory and repo root early in sys.path
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    # Try local import first
    from excel_helper import load_csv_to_dataframe, write_dataframe_to_excel, apply_steps, load_sheet_from_excel
except Exception:
    try:
        from tools.assistant.excel_helper import load_csv_to_dataframe, write_dataframe_to_excel, apply_steps, load_sheet_from_excel
    except Exception:
        load_csv_to_dataframe = None
        write_dataframe_to_excel = None
        apply_steps = None
        load_sheet_from_excel = None
try:
    from image_helper import image_to_text
except Exception:
    try:
        from tools.assistant.image_helper import image_to_text
    except Exception:
        image_to_text = None

    try:
        from whatsapp_helper import send_whatsapp
    except Exception:
        try:
            from tools.assistant.whatsapp_helper import send_whatsapp
        except Exception:
            send_whatsapp = None

    try:
        from email_helper import send_email_smtp, send_email_sendgrid
    except Exception:
        try:
            from tools.assistant.email_helper import send_email_smtp, send_email_sendgrid
        except Exception:
            send_email_smtp = None
            send_email_sendgrid = None

    try:
        from office_helper import create_text_file, create_docx, create_pptx
    except Exception:
        try:
            from tools.assistant.office_helper import create_text_file, create_docx, create_pptx
        except Exception:
            create_text_file = None
            create_docx = None
            create_pptx = None


def speak(text: str):
    """Speak the given text using pyttsx3 if available, otherwise print it."""
    print("Assistant:", text)
    if pyttsx3 is None:
        return
    try:
        engine = pyttsx3.init()
        # run in a separate thread to avoid blocking the REPL
        def _s():
            engine.say(text)
            engine.runAndWait()

        t = threading.Thread(target=_s, daemon=True)
        t.start()
    except Exception as e:
        print("TTS error:", e)


def get_openai_response(prompt: str) -> str:
    """Return a response from OpenAI ChatCompletion if configured, otherwise a simple echo reply.

    Uses a safe system prompt and falls back to the legacy completion echo when Chat is unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not _OPENAI_AVAILABLE:
        return f"(no OpenAI) Echo: {prompt}"

    try:
        openai.api_key = api_key
        # Prefer ChatCompletion (gpt-3.5-turbo style). Use a cautious system prompt.
        sys_prompt = (
            "You are a local assistant that helps manipulate CSV/Excel data and run commands only when explicitly confirmed. "
            "Never execute untrusted commands on your own. Be concise and helpful."
        )
        # Build a short chat exchange
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        # Use ChatCompletion when available
        if hasattr(openai, 'ChatCompletion'):
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=256)
            return resp.choices[0].message.content.strip()
        else:
            # Fallback to legacy completion endpoint
            resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=256)
            return resp.choices[0].text.strip()
    except Exception as e:
        return f"OpenAI error: {e}"


def execute_command(cmd: str) -> str:
    """Execute a shell command and return its combined stdout/stderr output.

    This runs the command with shell=True to preserve typical shell behavior on Windows and other OSes.
    """
    try:
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        out = completed.stdout or ""
        err = completed.stderr or ""
        result = out + ("\n" + err if err else "")
        return result.strip()
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as e:
        return f"Execution error: {e}"


def repl():
    speak("Local assistant ready. Type your message. Prefix commands with 'run: ' to execute them.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            speak("Goodbye")
            break

        if not user:
            continue

        if user.lower() in {"quit", "exit", "bye"}:
            speak("Goodbye")
            break

        # Command execution pattern: run: <shell command>
        if user.lower().startswith("run:") or user.lower().startswith("execute:"):
            parts = user.split(":", 1)
            if len(parts) < 2:
                speak("No command provided. Use: run: <command>")
                continue
            cmd = parts[1].strip()
            speak(f"You asked to run: {cmd}. Do you confirm? Type 'yes' to confirm.")
            confirm = input("Confirm (yes/no): ").strip().lower()
            if confirm in {"y", "yes"}:
                speak("Executing command now. Results will follow.")
                out = execute_command(cmd)
                if out:
                    speak("Command output follows.")
                    print(out)
                    # speak a concise summary
                    summary = out.splitlines()
                    speak(summary[0] if summary else "(no output)")
                else:
                    speak("Command executed with no output.")
            else:
                speak("Command canceled.")
            continue

        # Excel mini-REPL: start when user types 'excel' or 'excel:' prefix
        if user.lower().startswith("excel"):
            if load_csv_to_dataframe is None:
                speak("Excel support is not available (missing dependencies). Please install pandas and openpyxl.")
                continue
            speak("Entering Excel mode. Type 'help' for commands, 'exit' to leave Excel mode.")
            excel_mode()
            continue

        # Normal message: get response from OpenAI if available, otherwise echo
        reply = get_openai_response(user)
        speak(reply)

        # WhatsApp quick command
        if user.lower().startswith("whatsapp"):
            if send_whatsapp is None:
                speak("WhatsApp support is not available (missing dependencies).")
                continue
            # Expected format: whatsapp send to <number> message "<text>"
            import shlex
            try:
                parts = shlex.split(user)
                # find 'to' and 'message'
                if "to" in parts and "message" in parts:
                    idx_to = parts.index("to")
                    idx_msg = parts.index("message")
                    to = parts[idx_to + 1]
                    body = " ".join(parts[idx_msg + 1:])
                    speak(f"Send WhatsApp to {to}? Confirm? (yes/no)")
                    c = input("Confirm: ").strip().lower()
                    if c not in {"y", "yes"}:
                        speak("Canceled")
                        continue
                    out = send_whatsapp(to, body, dry_run=True)
                    speak(out)
                else:
                    speak("Invalid WhatsApp syntax. Use: whatsapp send to <number> message \"text\"")
            except Exception as e:
                speak(f"WhatsApp error: {e}")
            continue

        # Email quick command
        if user.lower().startswith("email"):
            if send_email_smtp is None and send_email_sendgrid is None:
                speak("Email support is not available (missing dependencies).")
                continue
            # Expected format: email send to <addr> subject "..." body "..."
            import shlex
            try:
                parts = shlex.split(user)
                if "to" in parts and "subject" in parts and "body" in parts:
                    idx_to = parts.index("to")
                    idx_subject = parts.index("subject")
                    idx_body = parts.index("body")
                    to = parts[idx_to + 1]
                    subject = parts[idx_subject + 1]
                    body = parts[idx_body + 1]
                    speak(f"Send email to {to} with subject '{subject}'? Confirm? (yes/no)")
                    c = input("Confirm: ").strip().lower()
                    if c not in {"y", "yes"}:
                        speak("Canceled")
                        continue
                    # Prefer SendGrid if key present
                    sg_key = os.environ.get("SENDGRID_API_KEY")
                    if sg_key and send_email_sendgrid is not None:
                        out = send_email_sendgrid(sg_key, os.environ.get("EMAIL_FROM", "noreply@example.com"), to, subject, body, dry_run=True)
                    else:
                        # attempt SMTP using env vars
                        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
                        smtp_port = int(os.environ.get("SMTP_PORT", 587))
                        smtp_user = os.environ.get("SMTP_USER")
                        smtp_pass = os.environ.get("SMTP_PASS")
                        out = send_email_smtp(smtp_host, smtp_port, smtp_user, smtp_pass, to, subject, body, dry_run=True)
                    speak(out)
                else:
                    speak("Invalid email syntax. Use: email send to <addr> subject \"...\" body \"...\"")
            except Exception as e:
                speak(f"Email error: {e}")
            continue

        # Office commands
        if user.lower().startswith("office") or user.lower().startswith("ppt") or user.lower().startswith("doc") or user.lower().startswith("note"):
            # Simple handlers:
            # note create <path> text "..."
            import shlex
            try:
                parts = shlex.split(user)
                if parts[0].lower() == "note" and parts[1].lower() == "create":
                    path = parts[2]
                    text = " ".join(parts[3:])
                    speak(f"Create text file {path}? Confirm? (yes/no)")
                    c = input("Confirm: ").strip().lower()
                    if c not in {"y","yes"}:
                        speak("Canceled")
                        continue
                    if create_text_file is None:
                        speak("Text helper not available.")
                        continue
                    out = create_text_file(path, text, dry_run=True)
                    speak(out)
                    continue
                if parts[0].lower() in {"ppt","pptx","presentation"} and parts[1].lower() == "create":
                    path = parts[2]
                    # simple single slide body after 'slide'
                    title = None
                    body = None
                    if "title" in parts:
                        title = parts[parts.index("title") + 1]
                    if "slide" in parts:
                        body = parts[parts.index("slide") + 1]
                    speak(f"Create pptx {path}? Confirm? (yes/no)")
                    c = input("Confirm: ").strip().lower()
                    if c not in {"y","yes"}:
                        speak("Canceled")
                        continue
                    if create_pptx is None:
                        speak("PPT helper not available.")
                        continue
                    slides = [{"title": title, "body": body}]
                    out = create_pptx(path, slides, dry_run=True)
                    speak(out)
                    continue
                if parts[0].lower() in {"doc","docx","word"} and parts[1].lower() == "create":
                    path = parts[2]
                    paragraphs = parts[3:]
                    speak(f"Create docx {path}? Confirm? (yes/no)")
                    c = input("Confirm: ").strip().lower()
                    if c not in {"y","yes"}:
                        speak("Canceled")
                        continue
                    if create_docx is None:
                        speak("DOCX helper not available.")
                        continue
                    out = create_docx(path, paragraphs, dry_run=True)
                    speak(out)
                    continue
            except Exception as e:
                speak(f"Office helper error: {e}")
            continue


def excel_mode():
    """A small interactive mini-REPL to operate on CSV/Excel files.

    Commands supported (interactive):
    - load csv <path> into <excel_path> sheet <sheet_name>
    - open excel <excel_path> sheet <sheet_name>
    - step "Expr" file <excel_path> sheet <sheet_name>    # apply expression like: Total = Price * Qty
    - save file <excel_path> sheet <sheet_name>           # saves last loaded dataframe
    - help
    - exit
    """
    current_df = None
    current_file = None
    current_sheet = None
    from tabulate import tabulate
    while True:
        try:
            cmd = input("Excel> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            speak("Leaving Excel mode")
            break
        if not cmd:
            continue
        lcmd = cmd.lower()
        if lcmd in {"exit", "quit"}:
            speak("Leaving Excel mode")
            break
        if lcmd == "help":
            print("Commands:\n - load csv <csv_path> into <excel_path> sheet <sheet_name>\n - open excel <excel_path> sheet <sheet_name>\n - step \"Expr\" file <excel_path> sheet <sheet_name>\n - save file <excel_path> sheet <sheet_name>\n - exit")
            continue
        # load csv
        if lcmd.startswith("load csv ") and " into " in lcmd and " sheet " in lcmd:
            # parse using shlex to support quoted paths
            import shlex
            try:
                parts = shlex.split(cmd)
                # format: load csv <csv_path> into <excel_path> sheet <sheet_name>
                idx_into = parts.index("into")
                idx_sheet = parts.index("sheet")
                csv_path = " ".join(parts[2:idx_into])
                excel_path = " ".join(parts[idx_into + 1:idx_sheet])
                sheet_name = " ".join(parts[idx_sheet + 1:])
                speak(f"Load CSV '{csv_path}' and write to '{excel_path}' sheet '{sheet_name}'. Preview first? (yes/no)")
                preview_confirm = input("Preview before saving (yes/no): ").strip().lower()
                df = load_csv_to_dataframe(csv_path)
                if preview_confirm in {"y", "yes"}:
                    # pretty-print head with tabulate and show dtypes
                    print(tabulate(df.head(), headers="keys", tablefmt="github", showindex=False))
                    print("\nColumn dtypes:")
                    print(df.dtypes)
                    speak("Preview shown. Confirm write? (yes/no)")
                else:
                    speak("Confirm write? (yes/no)")
                c = input("Confirm: ").strip().lower()
                if c not in {"y", "yes"}:
                    speak("Canceled")
                    continue
                write_dataframe_to_excel(df, excel_path, sheet_name, mode="w")
                speak("Saved CSV to Excel file")
            except Exception as e:
                speak(f"Error: {e}")
            continue
        # open excel
        if lcmd.startswith("open excel ") and " sheet " in lcmd:
            import shlex
            try:
                parts = shlex.split(cmd)
                idx_sheet = parts.index("sheet")
                excel_path = " ".join(parts[2:idx_sheet])
                sheet_name = " ".join(parts[idx_sheet + 1:])
                df = load_sheet_from_excel(excel_path, sheet_name)
                current_df = df
                current_file = excel_path
                current_sheet = sheet_name
                speak(f"Loaded sheet '{sheet_name}' from '{excel_path}'. Rows: {len(df)} Columns: {list(df.columns)[:5]}")
            except Exception as e:
                speak(f"Error: {e}")
            continue
        # step: apply calculation
        if lcmd.startswith("step ") and " file " in lcmd and " sheet " in lcmd:
            # step "Expr" file <excel_path> sheet <sheet_name>
            import re, shlex
            try:
                # parse using regex for the quoted expression, then shlex for remaining parts
                m = re.match(r'step\s+"([^"]+)"\s+file\s+(.+)\s+sheet\s+(.+)', cmd, flags=re.I)
                if not m:
                    speak("Invalid step syntax. Use: step \"Expr\" file <excel_path> sheet <sheet_name>")
                    continue
                expr = m.group(1)
                excel_path = m.group(2).strip()
                sheet_name = m.group(3).strip()
                # Preview option: show top rows after applying step without writing
                speak(f"Preview changes before writing? (yes/no)")
                pre = input("Preview (yes/no): ").strip().lower()
                df = load_sheet_from_excel(excel_path, sheet_name)
                if pre in {"y", "yes"}:
                    from excel_helper import preview_apply_steps
                    preview = preview_apply_steps(df, [expr], n=5)
                    try:
                        from tabulate import tabulate
                        print(tabulate(preview, headers="keys", tablefmt="github", showindex=False))
                    except Exception:
                        print(preview)
                    speak("Preview shown. Confirm write? (yes/no)")
                else:
                    speak(f"Apply step '{expr}' to file '{excel_path}' sheet '{sheet_name}'. Confirm? (yes/no)")
                c = input("Confirm: ").strip().lower()
                if c not in {"y", "yes"}:
                    speak("Canceled")
                    continue
                df = apply_steps(df, [expr])
                # write back to same sheet
                write_dataframe_to_excel(df, excel_path, sheet_name, mode="w")
                speak("Step applied and sheet updated.")
            except Exception as e:
                speak(f"Error: {e}")
            continue
        # save (save current_df)
        if lcmd.startswith("save file ") and " sheet " in lcmd:
            try:
                parts = cmd.split()
                idx_sheet = parts.index("sheet")
                excel_path = " ".join(parts[2:idx_sheet])
                sheet_name = " ".join(parts[idx_sheet + 1:])
                if current_df is None:
                    speak("No current DataFrame loaded. Use 'open excel' or 'load csv' first.")
                    continue
                speak(f"Save current data to '{excel_path}' sheet '{sheet_name}'? Confirm? (yes/no)")
                c = input("Confirm: ").strip().lower()
                if c not in {"y", "yes"}:
                    speak("Canceled")
                    continue
                write_dataframe_to_excel(current_df, excel_path, sheet_name, mode="w")
                speak("Saved current data to Excel file.")
            except Exception as e:
                speak(f"Error: {e}")
            continue

        speak("Unknown excel command. Type 'help' for commands.")


def image_mode():
    """Interactive image mode for OCR operations.

    Commands:
    - read image <path> [lang <tess_lang>]
    - save text <path>    # save last OCR text to file
    - exit
    """
    last_text = None
    while True:
        try:
            cmd = input("Image> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            speak("Leaving image mode")
            break
        if not cmd:
            continue
        lcmd = cmd.lower()
        if lcmd in {"exit", "quit"}:
            speak("Leaving image mode")
            break
        if lcmd.startswith("read image "):
            if image_to_text is None:
                speak("OCR support is not available (missing dependencies). Install pillow and pytesseract.")
                continue
            parts = cmd.split()
            # read image <path> [lang <lang>]
            try:
                path = parts[2]
                lang = None
                if "lang" in parts:
                    idx = parts.index("lang")
                    if idx + 1 < len(parts):
                        lang = parts[idx + 1]
                speak(f"Extract text from image '{path}'? Confirm? (yes/no)")
                c = input("Confirm: ").strip().lower()
                if c not in {"y", "yes"}:
                    speak("Canceled")
                    continue
                text = image_to_text(path, lang=lang)
                last_text = text
                speak("OCR complete. Showing first 2 lines:")
                for i, line in enumerate(text.splitlines()[:2]):
                    print(line)
                speak((text.splitlines()[0] if text.splitlines() else "(no text)") )
            except Exception as e:
                speak(f"OCR error: {e}")
            continue
        if lcmd.startswith("save text "):
            parts = cmd.split()
            if len(parts) < 3:
                speak("Usage: save text <path>")
                continue
            if last_text is None:
                speak("No OCR text available. Use 'read image' first.")
                continue
            path = " ".join(parts[2:])
            speak(f"Save OCR text to '{path}'? Confirm? (yes/no)")
            c = input("Confirm: ").strip().lower()
            if c not in {"y", "yes"}:
                speak("Canceled")
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(last_text)
            speak("Saved OCR text to file.")
            continue
        speak("Unknown image command. Use 'read image <path>' or 'save text <path>'")


if __name__ == "__main__":
    if pyttsx3 is None:
        print("Warning: 'pyttsx3' is not installed. Speech output will be disabled.")
    if _OPENAI_AVAILABLE and not os.environ.get("OPENAI_API_KEY"):
        print("OpenAI SDK present but OPENAI_API_KEY not set — OpenAI calls disabled.")
    repl()
