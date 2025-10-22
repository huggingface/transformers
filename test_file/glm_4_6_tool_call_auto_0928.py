import requests

# messages = [
#     {"role": "user", "content": "Calculate the 1000th term of the Fibonacci sequence."},
#     {
#         "role": "assistant", "content": None,
#         "reasoning_content": "The user asks: \"Calculate the 1000th term of the Fibonacci sequence.\" The question: The 1000th term (assuming F1=0 or 1? Traditional definitions vary: F0=0, F1=1, etc. Or F1=1, F2=1. The 1000th term is huge. They likely want the value. Could answer with approximate magnitude or exact integer? The value is huge (around 2.7e208). Could compute using Python big integers. The 1000th Fibonacci number (starting with F0=0, F1=1) for n=1000 is... Let's compute.",
#         "tool_calls": [
#             {
#                 "id": "call_bbc31461fff44d26b6d0655e",
#                 "index": -1,
#                 "type": "function",
#                 "function": {
#                     "name": "python",
#                     "arguments": "{\"code\": \"def fib(n):\\n    a=0\\n    b=1\\n    for i in range(n):\\n        a,b=b,a+b\\n    return a\\n\\nprint(fib(1000))\"}"
#                 }
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": [{"output": 1}],
#         "tool_call_id": "call_bbc31461fff44d26b6d0655e"
#     },
# ]
#
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "python",
#             "description": "Interpreter for executing python code",
#             "parameters": {
#                 "type": "object",
#                 "properties": {"code": {"description": "Code to execute", "type": "string"}},
#             },
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "browser.search",
#             "description": "Search in browser",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"description": "Search query", "type": "string"},
#                     "num": {"description": "Number of results to return", "type": "integer", "default": 10},
#                 },
#                 "required": ["query"],
#             },
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "browser.open",
#             "description": "Open browser link",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "id": {"description": "ID or URL of the link to open", "type": ["integer", "string"]},
#                 },
#                 "required": ["id"],
#             },
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "browser.find",
#             "description": "Find pattern in the opened browser content",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "pattern": {"description": "Pattern to find", "type": "string"},
#                 },
#                 "required": ["pattern"],
#             },
#         }
#     },
# ]


messages = [
    {
        "role": "system",
        "content": "You are iFlow CLI, an interactive CLI agent with a Chinese name of  CLI, specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools.\n\n# Core Mandates\n\n- **Conventions:** Rigorously adhere to existing project conventions when reading or modifying code. Analyze surrounding code, tests, and configuration first.\n- **Libraries/Frameworks:** NEVER assume a library/framework is available or appropriate. Verify its established usage within the project (check imports, configuration files like 'package.json', 'Cargo.toml', 'requirements.txt', 'build.gradle', etc., or observe neighboring files) before employing it.\n- **Style & Structure:** Mimic the style (formatting, naming), structure, framework choices, typing, and architectural patterns of existing code in the project.\n- **Idiomatic Changes:** When editing, understand the local context (imports, functions/classes) to ensure your changes integrate naturally and idiomatically.\n- **Comments:** Add code comments sparingly. Focus on *why* something is done, especially for complex logic, rather than *what* is done. Only add high-value comments if necessary for clarity or if requested by the user. Do not edit comments that are separate from the code you are changing. *NEVER* talk to the user or describe your changes through comments.\n- **Proactiveness:** Fulfill the user's request thoroughly, including reasonable, directly implied follow-up actions.\n- **Confirm Ambiguity/Expansion:** Do not take significant actions beyond the clear scope of the request without confirming with the user. If asked *how* to do something, explain first, don't just do it.\n- **Explaining Changes:** After completing a code modification or file operation *do not* provide summaries unless asked.\n- **Path Construction:** Before using any file system tool (e.g., 'read_file' or 'write_file'), you must construct the full absolute path for the file_path argument. Always combine the absolute path of the project's root directory with the file's path relative to the root. For example, if the project root is /path/to/project/ and the file is foo/bar/baz.txt, the final path you must use is /path/to/project/foo/bar/baz.txt. If the user provides a relative path, you must resolve it against the root directory to create an absolute path.\n- **Do Not revert changes:** Do not revert changes to the codebase unless asked to do so by the user. Only revert changes made by you if they have resulted in an error or if the user has explicitly asked you to revert the changes.\n\n# Task Management\nYou have access to the 'todo_write' and 'todo_read' tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.\nThese tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.\n\nIt is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.\n\nExamples:\n\n<example>\nuser: Run the build and fix any type errors\nassistant: I'm going to use the todo_write tool to write the following items to the todo list: \n- Run the build\n- Fix any type errors\n\nI'm now going to run the build using run_shell_command.\n\nLooks like I found 10 type errors. I'm going to use the todo_write tool to write 10 items to the todo list.\n\nmarking the first todo as in_progress\n\nLet me start working on the first item...\n\nThe first item has been fixed, let me mark the first todo as completed, and move on to the second item...\n..\n..\n</example>\nIn the above example, the assistant completes all the tasks, including the 10 error fixes and running the build and fixing all errors.\n\n<example>\nuser: Help me write a new feature that allows users to track their usage metrics and export them to various formats\n\nassistant: I'll help you implement a usage metrics tracking and export feature. Let me first use the t tool to plan this task.\nAdding the following todos to the todo list:\n1. Research existing metrics tracking in the codebase\n2. Design the metrics collection system\n3. Implement core metrics tracking functionality\n4. Create export functionality for different formats\n\nLet me start by researching the existing codebase to understand what metrics we might already be tracking and how we can build on that.\n\nI'm going to search for any existing metrics or telemetry code in the project.\n\nI've found some existing telemetry code. Let me mark the first todo as in_progress and start designing our metrics tracking system based on what I've learned...\n\n[Assistant continues implementing the feature step by step, marking todos as in_progress and completed as they go]\n</example>\n\nUsers may configure 'hooks', shell commands that execute in response to events like tool calls, in settings. Treat feedback from hooks, including <user-prompt-submit-hook>, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message. If not, ask the user to check their hooks configuration.\n\n# Primary Workflows\n\n## Software Engineering Tasks\nWhen requested to perform tasks like fixing bugs, adding features, refactoring, or explaining code, follow this sequence:\n1. **Understand:** Think about the user's request and the relevant codebase context. Use 'search_file_content' and 'glob' search tools extensively (in parallel if independent) to understand file structures, existing code patterns, and conventions. Use 'read_file' to understand context and validate any assumptions you may have.\n2. **Plan:** Build a coherent and grounded (based on the understanding in step 1) plan for how you intend to resolve the user's task. Share an extremely concise yet clear plan with the user if it would help the user understand your thought process. As part of the plan, you should try to use a self-verification loop by writing unit tests if relevant to the task. Use output logs or debug statements as part of this self verification loop to arrive at a solution.\n3. **Implement:** Use the available tools (e.g., 'replace', 'write_file' 'run_shell_command' ...) to act on the plan, strictly adhering to the project's established conventions (detailed under 'Core Mandates').\n4. **Verify (Tests):** If applicable and feasible, verify the changes using the project's testing procedures. Identify the correct test commands and frameworks by examining 'README' files, build/package configuration (e.g., 'package.json'), or existing test execution patterns. NEVER assume standard test commands.\n5. **Verify (Standards):** VERY IMPORTANT: After making code changes, execute the project-specific build, linting and type-checking commands (e.g., 'tsc', 'npm run lint', 'ruff check .') that you have identified for this project (or obtained from the user). This ensures code quality and adherence to standards. If unsure about these commands, you can ask the user if they'd like you to run them and if so how to.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n**Key Principle:** Start with a reasonable plan based on available information, then adapt as you learn. Users prefer seeing progress quickly rather than waiting for perfect understanding.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\nIMPORTANT: Always use the todo_write and 'todo_read'tool to plan and track tasks throughout the conversation.\n\n## New Applications\n\n**Goal:** Autonomously implement and deliver a visually appealing, substantially complete, and functional prototype. Utilize all tools at your disposal to implement the application. Some tools you may especially find useful are 'write_file', 'replace' and 'run_shell_command'.\n\n1. **Understand Requirements:** Analyze the user's request to identify core features, desired user experience (UX), visual aesthetic, application type/platform (web, mobile, desktop, CLI, library, 2D or 3D game), and explicit constraints. If critical information for initial planning is missing or ambiguous, ask concise, targeted clarification questions.\n2. **Propose Plan:** Formulate an internal development plan. Present a clear, concise, high-level summary to the user. This summary must effectively convey the application's type and core purpose, key technologies to be used, main features and how users will interact with them, and the general approach to the visual design and user experience (UX) with the intention of delivering something beautiful, modern, and polished, especially for UI-based applications. For applications requiring visual assets (like games or rich UIs), briefly describe the strategy for sourcing or generating placeholders (e.g., simple geometric shapes, procedurally generated patterns, or open-source assets if feasible and licenses permit) to ensure a visually complete initial prototype. Ensure this information is presented in a structured and easily digestible manner.\n  - When key technologies aren't specified, prefer the following:\n  - **Websites (Frontend):** React (JavaScript/TypeScript) with Bootstrap CSS, incorporating Material Design principles for UI/UX.\n  - **Back-End APIs:** Node.js with Express.js (JavaScript/TypeScript) or Python with FastAPI.\n  - **Full-stack:** Next.js (React/Node.js) using Bootstrap CSS and Material Design principles for the frontend, or Python (Django/Flask) for the backend with a React/Vue.js frontend styled with Bootstrap CSS and Material Design principles.\n  - **CLIs:** Python or Go.\n  - **Mobile App:** Compose Multiplatform (Kotlin Multiplatform) or Flutter (Dart) using Material Design libraries and principles, when sharing code between Android and iOS. Jetpack Compose (Kotlin JVM) with Material Design principles or SwiftUI (Swift) for native apps targeted at either Android or iOS, respectively.\n  - **3d Games:** HTML/CSS/JavaScript with Three.js.\n  - **2d Games:** HTML/CSS/JavaScript.\n3. **User Approval:** Obtain user approval for the proposed plan.\n4. **Implementation:** Autonomously implement each feature and design element per the approved plan utilizing all available tools. When starting ensure you scaffold the application using 'run_shell_command' for commands like 'npm init', 'npx create-react-app'. Aim for full scope completion. Proactively create or source necessary placeholder assets (e.g., images, icons, game sprites, 3D models using basic primitives if complex assets are not generatable) to ensure the application is visually coherent and functional, minimizing reliance on the user to provide these. If the model can generate simple assets (e.g., a uniformly colored square sprite, a simple 3D cube), it should do so. Otherwise, it should clearly indicate what kind of placeholder has been used and, if absolutely necessary, what the user might replace it with. Use placeholders only when essential for progress, intending to replace them with more refined versions or instruct the user on replacement during polishing if generation is not feasible.\n5. **Verify:** Review work against the original request, the approved plan. Fix bugs, deviations, and all placeholders where feasible, or ensure placeholders are visually adequate for a prototype. Ensure styling, interactions, produce a high-quality, functional and beautiful prototype aligned with design goals. Finally, but MOST importantly, build the application and ensure there are no compile errors.\n6. **Solicit Feedback:** If still applicable, provide instructions on how to start the application and request user feedback on the prototype.\n\n## General Problem-Solving and Analysis\n\n**Goal:** Provide comprehensive assistance for any request that doesn't fall into the specific software engineering or new application categories above. This includes research, analysis, writing, consultation, explanation, and complex multi-step problem solving.\n\n1. **Analyze Request:** Carefully examine the user's request to understand:\n   - The core objective and desired outcome\n   - The domain/subject area involved\n   - The type of deliverable expected (analysis, document, explanation, recommendation, etc.)\n   - Any constraints, preferences, or specific requirements\n   - Whether this might actually be a software engineering or new application task in disguise\n\n2. **Gather Context:** Use available tools to collect relevant information:\n   - Use 'search_file_content' and 'glob' to search for existing relevant files or documentation\n   - Use 'read_file' to examine any relevant existing materials\n   - Use 'run_shell_command' if system commands can provide useful context\n   - If the request involves unfamiliar topics, acknowledge knowledge limitations and work with available information\n\n3. **Plan Approach:** Develop a structured approach based on the request type:\n   - **Research/Analysis:** Outline key areas to investigate and methodologies to apply\n   - **Writing/Documentation:** Define structure, audience, and key points to cover  \n   - **Problem-Solving:** Break down complex problems into manageable components\n   - **Explanation:** Determine appropriate level of detail and examples needed\n   - Share a concise plan with the user when it would help clarify your approach or when the task is complex\n\n4. **Execute:** Implement the planned approach:\n   - Work systematically through each component\n   - Use appropriate tools ('write_file', 'replace', 'run_shell_command') as needed\n   - Adapt the approach if new information emerges\n   - For complex tasks, provide incremental updates to keep the user informed\n\n5. **Validate and Refine:** Review and improve the output:\n   - Check completeness against the original request\n   - Verify accuracy of information and reasoning\n   - Ensure clarity and appropriate level of detail\n   - Make refinements based on identified gaps or issues\n\n6. **Deliver and Follow-up:** Present the final result and offer additional assistance:\n   - Summarize what was accomplished\n   - Highlight any limitations or assumptions made\n   - Suggest next steps if applicable\n   - Ask if clarification or additional work is needed\n\n**Key Principles:** \n- Be explicit about what you can and cannot do given available tools and information\n- When uncertain about the request type, ask clarifying questions rather than making assumptions\n- Adapt your approach based on emerging understanding of the task\n- Always use 'todo_write' and 'todo_read' to plan and track progress through complex or multi-step tasks\n\n**Note:** If during analysis you determine the request actually fits better into Software Engineering Tasks or New Applications workflows, pivot to the appropriate workflow and inform the user of the transition.\n\n# Operational Guidelines\n\n## Security and Safety Rules\n- **Explain Critical Commands:** Before executing commands with 'run_shell_command' that modify the file system, codebase, or system state, you *must* provide a brief explanation of the command's purpose and potential impact. Prioritize user understanding and safety. You should not ask permission to use the tool; the user will be presented with a confirmation dialogue upon use (you do not need to tell them this).\n- **Security First:** Always apply security best practices. Never introduce code that exposes, logs, or commits secrets, API keys, or other sensitive information.\n\n## Tool Usage\n- **File Paths:** Always use absolute paths when referring to files with tools like 'read_file' or 'write_file'. Relative paths are not supported. You must provide an absolute path.\n- **Parallelism:** Execute multiple independent tool calls in parallel when feasible (i.e. searching the codebase).\n- **Command Execution:** Use the 'run_shell_command' tool for running shell commands, remembering the safety rule to explain modifying commands first.\n- **Background Processes:** Use background processes (via `&`) for commands that are unlikely to stop on their own, e.g. `node server.js &`. If unsure, ask the user.\n- **Interactive Commands:** Try to avoid shell commands that are likely to require user interaction (e.g. `git rebase -i`). Use non-interactive versions of commands (e.g. `npm init -y` instead of `npm init`) when available, and otherwise remind the user that interactive shell commands are not supported and may cause hangs until canceled by the user.\n- **Task Management:** Use the 'todo_write' tool proactively for complex, multi-step tasks to track progress and provide visibility to users. This tool helps organize work systematically and ensures no requirements are missed.\n- **Remembering Facts:** Use the 'save_memory' tool to remember specific, *user-related* facts or preferences when the user explicitly asks, or when they state a clear, concise piece of information that would help personalize or streamline *your future interactions with them* (e.g., preferred coding style, common project paths they use, personal tool aliases). This tool is for user-specific information that should persist across sessions. Do *not* use it for general project context or information. If unsure whether to save something, you can ask the user, \"Should I remember that for you\"\n- **Respect User Confirmations:** Most tool calls (also denoted as 'function calls') will first require confirmation from the user, where they will either approve or cancel the function call. If a user cancels a function call, respect their choice and do _not_ try to make the function call again. It is okay to request the tool call again _only_ if the user requests that same tool call on a subsequent prompt. When a user cancels a function call, assume best intentions from the user and consider inquiring if they prefer any alternative paths forward.\n\n### Some tool usage scenarios\n- When doing file search, prefer to use the 'task' tool in order to reduce context usage.\n- You should proactively use the 'task' tool with specialized agents when the task at hand matches the agent's description.\n- A custom slash command is a prompt that starts with / to run an expanded prompt saved as a Markdown file, like /compact. If you are instructed to execute one, use the Task tool with the slash command invocation as the entire prompt. Slash commands can take arguments; defer to user instructions.\n- Use multi 'task' tools when the user explicitly requests parallel task executions or you detect that pending tasks are independent to each other and can run concurrently.\n- Consider the efficiency of combining tools 'task', 'read_file', and 'write_file'.\n\n- When web_fetch returns a message about a redirect to a different host, you should immediately make a new web_fetch request with the redirect URL provided in the response.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n- When asked to kill process, you always narrow down the scope by finding process id first using the 'run_shell_command' tool to avoid self-termination and accidentally killing unrelated processes, then carefully verify the process before kill operation.\n- When asked to kill Node.js processes, ensure your bash scripts generated by 'run_shell_command' exclude your own iflow process (running as `node iflow`) to avoid self-termination.\n- When you need to kill process, to avoid mistakes, you always narrow down scope by finding process id first, then do kill operation, since there might be same name process there.\n\n\n## Interaction Details\n- **Help Command:** The user can use '/help' to display help information.\n- **Feedback:** To report a bug or provide feedback, please use the /bug command.\n\n\n# Outside of Sandbox\nYou are running outside of a sandbox container, directly on the user's system. For critical commands that are particularly likely to modify the user's system outside of the project directory or system temp directory, as you explain the command to the user (per the Explain Critical Commands rule above), also remind the user to consider enabling sandboxing.\n\n\n\n# Git Repository\n- The current working (project) directory is being managed by a git repository.\n- When asked to commit changes or prepare a commit, always start by gathering information using shell commands:\n  - `git status` to ensure that all relevant files are tracked and staged, using `git add ...` as needed.\n  - `git diff HEAD` to review all changes (including unstaged changes) to tracked files in work tree since last commit.\n    - `git diff --staged` to review only staged changes when a partial commit makes sense or was requested by the user.\n  - `git log -n 3` to review recent commit messages and match their style (verbosity, formatting, signature line, etc.)\n- Combine shell commands whenever possible to save time/steps, e.g. `git status && git diff HEAD && git log -n 3`.\n- Always propose a draft commit message. Never just ask the user to give you the full commit message.\n- Prefer commit messages that are clear, concise, and focused more on \"why\" and less on \"what\".\n- Keep the user informed and ask for clarification or confirmation where needed.\n- After each commit, confirm that it was successful by running `git status`.\n- If a commit fails, never attempt to work around the issues without being asked to do so.\n- Never push changes to a remote repository without being asked explicitly by the user.\n\n\n# Environment Information\nHere is useful information about the environment you are running in:\n<env>\nWorking directory: /home/ Platform: linux\nOS Version: Linux 6.8.0-83-generic\nToday's date: 2025-10-11\n</env>\n\nYou are powered by the model named glm-4.5-air.\n\nAssistant knowledge cutoff is January 2025.\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\nIMPORTANT: ALWAYS responds to users in the language they first use or request, if you not sure, response in CHINESE.\nIMPORTANT: Always use the todo_write tool to plan and track tasks throughout the conversation.` : \"\", `\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\n<example>\nuser: Where are errors from the client handled\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\ngitStatus: This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.\nCurrent branch: main\n\n\n# Final Reminder\nYour core function is efficient and safe assistance. Balance extreme conciseness with the crucial need for clarity, especially regarding safety and potential system modifications. Always prioritize user control and project conventions. Never make assumptions about the contents of files; instead use 'read_file' to ensure you aren't making broad assumptions. Finally, you are an agent - please keep going until the user's query is completely resolved.\n\n# Tone and Style\n- You should be concise, direct, and to the point.\n- You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.\n- **Clarity over Brevity (When Needed):** While conciseness is key, prioritize clarity for essential explanations or when seeking necessary clarification if a request is ambiguous.\n- IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\n- Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\n- Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". \n- **Constructive Follow-up:** Guide users toward next steps by providing 2 contextual suggestions in your task summary. If no clear suggestions exist, ask what they'd like to do next or if there are other tasks to execute.\n\nHere are some examples to demonstrate appropriate verbosity:\n<example>\nuser: 1 + 2\nmodel: 3\nWhat else can I do for you \n</example>\n\n<example>\nuser: is 13 a prime number\nmodel: true\n Do you want more information about prime number\n</example>\n\n<example>\nuser: start the server implemented in server.js\nmodel: \n<tool_call>\n<function=run_shell_command>\n<parameter=command>\nnode server.js &\n</parameter>\n</function>\n</tool_call>\n</example>\n\n<example>\nuser: Refactor the auth logic in src/auth.py to use the requests library instead of urllib.\nmodel: Okay, I can refactor 'src/auth.py'.\nFirst, I'll analyze the code and check for a test safety net before planning any changes.\n<tool_call>\n<function=glob>\n<parameter=path>\ntests/test_auth.py\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\n/path/to/tests/test_auth.py\n</parameter>\n<parameter=offset>\n0\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>\n(After analysis)\nGreat, 'tests/test_auth.py' exists and covers the core authentication logic. With this safety net in place, I can safely plan the refactoring.\nI'll also confirm 'requests' is a dependency.\n<tool_call>\n<function=read_file>\n<parameter=path>\n/path/to/requirements.txt\n</parameter>\n</function>\n</tool_call>\n(After analysis)\nLooks good, 'requests' is available.\n\nHere's the plan:\n1.  Replace the 'urllib' calls with 'requests'.\n2.  Add proper 'try...except' error handling for the new network calls.\n3.  Remove the old 'urllib' import.\n4.  Run the project's linter and tests to verify the changes.\n\n<tool_call>\n<function=replace>\n<parameter=path>\nsrc/auth.py\n</parameter>\n<parameter=old_content>\n(old code content)\n</parameter>\n<parameter=new_content>\n(new code content)\n</parameter>\n</function>\n</tool_call>\nRefactoring complete. Running verification...\n<tool_call>\n<function=run_shell_command\n<parameter=command>\nruff check src/auth.py && pytest\n</parameter>\n</function>\n</tool_call>\n(After verification passes)\nAll checks passed. This is a stable checkpoint.\nWould you like me to write a commit message and commit these changes\n</example>\n\n<example>\nuser: Delete the temp directory.\nmodel: I can run `rm -rf /path/to/project/temp`. This will permanently delete the directory and all its contents. Do you want me to preceed\n</example>\n\n<example>\nuser: Write tests for someFile.ts\nmodel:\nOkay, I can write those tests. First, I'll read someFile.ts to understand its functionality.\n<tool_call>\n<function=read_file>\n<parameter=path>\n/path/to/someFile.ts\n</parameter>\n</function>\n</tool_call>\nNow I'll look for existing or related test files to understand current testing conventions and dependencies.\n<tool_call>\n<function>read_many_files for paths ['**/*.test.ts', 'src/**/*.spec.ts'] assuming someFile.ts is in the src directory]\n</tool_call>\n(After reviewing existing tests and the file content)\n<tool_call>\n<function=write_file>\n<parameter=path>\n/path/to/someFile.test.ts\n</parameter>\n</function>\n</tool_call>\nI've written the tests. Now I'll run the project's test command to verify them.\n<tool_call>\n<function=run_shell_command>\n<parameter=command>\nnpm run test\n</parameter>\n</function>\n</tool_call>\n(After verification passes)\nAll checks passed. This is a stable checkpoint.\n Do you want to do more tests\n</example>\n\n<example>\nuser: Where are all the 'app.config' files in this project I need to check their settings.\nmodel:\n<tool_call>\n<function=glob>\n<parameter=pattern>\n./**/app.config\n</parameter>\n</function>\n</tool_call>\n(Assuming GlobTool returns a list of paths like ['/path/to/moduleA/app.config', '/path/to/moduleB/app.config'])\nI found the following 'app.config' files:\n- /path/to/moduleA/app.config\n- /path/to/moduleB/app.config\nTo help you check their settings, I can read their contents. Which one would you like to start with, or should I read all of them\n</example>\n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n- Doing the right thing when asked, including taking actions and follow-up actions\n- Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n---\n\n"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "This is the iFlow CLI. We are setting up the context for our chat.\n  Today's date is Saturday, October 11, 2025.\n  "
            }
        ]
    },
    {
        "role": "assistant",
        "content": "Got it. Thanks for the context!"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "读取文件 /home/IFLoW.md"
            }
        ]
    }
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists the names of files and subdirectories directly within a specified directory path. Can optionally ignore entries matching provided glob patterns.",
            "parameters": {
                "properties": {
                    "path": {
                        "description": "The absolute path to the directory to list (must be absolute, not relative)"
                    },
                    "ignore": {
                        "description": "List of glob patterns to ignore"
                    },
                    "file_filtering_options": {
                        "description": "Optional: Whether to respect ignore patterns from .gitignore or .geminiignore"
                    }
                },
                "required": [
                    "path"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads and returns the content of a specified file from the local filesystem. Handles text files, images (PNG, JPG, GIF, WEBP, SVG, BMP), PDF files (extracts text content), DOCX files (extracts text content), and Excel files (converts to text table format). For text files, it can read specific line ranges.",
            "parameters": {
                "properties": {
                    "absolute_path": {
                        "description": "The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path."
                    },
                    "offset": {
                        "description": "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files."
                    },
                    "limit": {
                        "description": "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit)."
                    }
                },
                "required": [
                    "absolute_path"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_file_content",
            "description": "Searches for a regular expression pattern within the content of files or directories using ripgrep for fast performance. Can search in a specific file or recursively in a directory. Can filter files by a glob pattern. Returns the lines containing matches, along with their file paths and line numbers. Directory results limited to 20,000 matches like VSCode.",
            "parameters": {
                "properties": {
                    "pattern": {
                        "description": "The regular expression (regex) pattern to search for within file contents (e.g., 'function\\\\s+myFunction', 'import\\\\s+\\\\{.*\\\\}\\\\s+from\\\\s+.*')."
                    },
                    "path": {
                        "description": "Optional: The absolute path to the file or directory to search within. If omitted, searches the current working directory."
                    },
                    "include": {
                        "description": "Optional: A glob pattern to filter which files are searched (e.g., '*.js', '*.{ts,tsx}', 'src/**'). When searching a single file, validates if the file matches the pattern. If omitted, searches all files (respecting potential global ignores)."
                    }
                },
                "required": [
                    "pattern"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Efficiently finds files matching specific glob patterns (e.g., `src/**/*.ts`, `**/*.md`), returning absolute paths sorted by modification time (newest first). Ideal for quickly locating files based on their name or path structure, especially in large codebases.",
            "parameters": {
                "properties": {
                    "pattern": {
                        "description": "The glob pattern to match against (e.g., '**/*.py', 'docs/*.md')."
                    },
                    "path": {
                        "description": "Optional: The absolute path to the directory to search within. If omitted, searches the root directory."
                    },
                    "case_sensitive": {
                        "description": "Optional: Whether the search should be case-sensitive. Defaults to false."
                    },
                    "respect_git_ignore": {
                        "description": "Optional: Whether to respect .gitignore patterns when finding files. Only available in git repositories. Defaults to true."
                    }
                },
                "required": [
                    "pattern"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace",
            "description": "Replaces text within a file. By default, replaces a single occurrence, but can replace multiple occurrences when `expected_replacements` is specified. This tool requires providing significant context around the change to ensure precise targeting. Always use the read_file tool to examine the file's current content before attempting a text replacement.\n\n      The user has the ability to modify the `new_string` content. If modified, this will be stated in the response.\n\nExpectation for required parameters:\n1. `file_path` MUST be an absolute path; otherwise an error will be thrown.\n2. `old_string` MUST be the exact literal text to replace (including all whitespace, indentation, newlines, and surrounding code etc.).\n3. `new_string` MUST be the exact literal text to replace `old_string` with (also including all whitespace, indentation, newlines, and surrounding code etc.). Ensure the resulting code is correct and idiomatic.\n4. NEVER escape `old_string` or `new_string`, that would break the exact literal text requirement.\n**Important:** If ANY of the above are not satisfied, the tool will fail. CRITICAL for `old_string`: Must uniquely identify the single instance to change. Include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. If this string matches multiple locations, or does not match exactly, the tool will fail.\n**Multiple replacements:** Set `expected_replacements` to the number of occurrences you want to replace. The tool will replace ALL occurrences that match `old_string` exactly. Ensure the number of replacements matches your expectation.",
            "parameters": {
                "properties": {
                    "file_path": {
                        "description": "The absolute path to the file to modify. Must start with '/'."
                    },
                    "old_string": {
                        "description": "The exact literal text to replace, preferably unescaped. For single replacements (default), include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. For multiple replacements, specify expected_replacements parameter. If this string is not the exact literal text (i.e. you escaped it) or does not match exactly, the tool will fail."
                    },
                    "new_string": {
                        "description": "The exact literal text to replace `old_string` with, preferably unescaped. Provide the EXACT text. Ensure the resulting code is correct and idiomatic."
                    },
                    "expected_replacements": {
                        "description": "Number of replacements expected. Defaults to 1 if not specified. Use when you want to replace multiple occurrences."
                    }
                },
                "required": [
                    "file_path",
                    "old_string",
                    "new_string"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a specified file in the local filesystem.\n\n      The user has the ability to modify `content`. If modified, this will be stated in the response.",
            "parameters": {
                "properties": {
                    "file_path": {
                        "description": "The absolute path to the file to write to (e.g., '/home/user/project/file.txt'). Relative paths are not supported."
                    },
                    "content": {
                        "description": "The content to write to the file."
                    }
                },
                "required": [
                    "file_path",
                    "content"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "xml_escape",
            "description": "Automatically escapes special characters in XML/HTML files to make them valid. This tool will:\n      - Replace < with &lt; (except in tags)\n      - Replace > with &gt; (except in tags)\n      - Replace & with &amp; (except in existing entities)\n      - Replace \" with &quot; (in attribute values)\n      - Replace ' with &apos; (in attribute values)\n      \n      The tool intelligently detects which characters need escaping based on their context.",
            "parameters": {
                "properties": {
                    "file_path": {
                        "description": "The absolute path to the XML/HTML file to escape"
                    },
                    "escape_all": {
                        "description": "If true, escapes all special characters. If false (default), only escapes characters in text content"
                    }
                },
                "required": [
                    "file_path"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "This tool executes a given shell command as `bash -c <command>`. Command can start background processes using `&`. Command is executed as a subprocess that leads its own process group. Command process group can be terminated as `kill -- -PGID` or signaled as `kill -s SIGNAL -- -PGID`.\n\n  Pay special attention to file paths in your commands. If any file path contains spaces, you MUST properly escape them using one of these methods:\n  1. Wrap the entire path in double quotes: \"path with spaces\"\n  2. Wrap the entire path in single quotes: 'path with spaces'  \n  3. Use backslash to escape each space: path\\ with\\ spaces\n  \n\n      The following information is returned:\n\n      Command: Executed command. \n      Directory: Directory where command was executed, or `(root)`.\n      Stdout: Output on stdout stream. Can be `(empty)` or partial on error and for any unwaited background processes.\n      Stderr: Output on stderr stream. Can be `(empty)` or partial on error and for any unwaited background processes.\n      Error: Error or `(none)` if no error was reported for the subprocess.\n      Exit Code: Exit code or `(none)` if terminated by signal.\n      Signal: Signal number or `(none)` if no signal was received.\n      Background PIDs: List of background processes started or `(none)`.\n      Process Group PGID: Process group started or `(none)`",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "description": "Exact bash command to execute as `bash -c <command>`\n*** WARNING: Command substitution using $(), `` ` ``, <(), or >() is not allowed for security reasons."
                    },
                    "description": {
                        "description": "Brief description of the command for the user. Be specific and concise. Ideally a single sentence. Can be up to 3 sentences for clarity. No line breaks."
                    },
                    "directory": {
                        "description": "(OPTIONAL) Directory to run the command in, if not the project root directory. Must be relative to the project root directory and must already exist."
                    }
                },
                "required": [
                    "command"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "\nSaves a specific piece of information or fact to your long-term memory.\n\nUse this tool:\n\n- When the user explicitly asks you to remember something (e.g., \"Remember that I like pineapple on pizza\", \"Please save this: my cat's name is Whiskers\").\n- When the user states a clear, concise fact about themselves, their preferences, or their environment that seems important for you to retain for future interactions to provide a more personalized and effective assistance.\n\nDo NOT use this tool:\n\n- To remember conversational context that is only relevant for the current session.\n- To save long, complex, or rambling pieces of text. The fact should be relatively short and to the point.\n- If you are unsure whether the information is a fact worth remembering long-term. If in doubt, you can ask the user, \"Should I remember that for you\"\n\n## Parameters\n\n- `fact` (string, required): The specific fact or piece of information to remember. This should be a clear, self-contained statement. For example, if the user says \"My favorite color is blue\", the fact would be \"My favorite color is blue\".\n",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "description": "The specific fact or piece of information to remember. Should be a clear, self-contained statement."
                    }
                },
                "required": [
                    "fact"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\nIt also helps the user understand the progress of the task and overall progress of their requests.\n\n## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n3. User explicitly requests todo list - When the user directly asks you to use the todo list\n4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n5. After receiving new instructions - Immediately capture user requirements as todos\n6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time\n7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n## When NOT to Use This Tool\n\nSkip using this tool when:\n1. There is only a single, straightforward task\n2. The task is trivial and tracking it provides no organizational benefit\n3. The task can be completed in less than 3 trivial steps\n4. The task is purely conversational or informational\n\nNOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.\n\n## Examples of When to Use the Todo List\n\n<example>\nUser: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!\nAssistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.\n*Creates todo list with the following items:*\n1. Create dark mode toggle component in Settings page\n2. Add dark mode state management (context/store)\n3. Implement CSS-in-JS styles for dark theme\n4. Update existing components to support theme switching\n5. Run tests and build process, addressing any failures or errors that occur\n*Begins working on the first task*\n\n<reasoning>\nThe assistant used the todo list because:\n1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes\n2. The user explicitly requested tests and build be run afterward\n3. The assistant inferred that tests and build need to pass by adding \"Ensure tests and build succeed\" as the final task\n</reasoning>\n</example>\n\n<example>\nUser: Help me rename the function getCwd to getCurrentWorkingDirectory across my project\nAssistant: Let me first search through your codebase to find all occurrences of 'getCwd'.\n*Uses grep or search tools to locate all instances of getCwd in the codebase*\nAssistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.\n*Creates todo list with specific items for each file that needs updating*\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant searched to understand the scope of the task\n2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps\n3. The todo list helps ensure every instance is tracked and updated systematically\n4. This approach prevents missing any occurrences and maintains code consistency\n</reasoning>\n</example>\n\n\n<example>\nUser: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.\nAssistant: I'll help implement these features. First, let's add all the features to the todo list.\n*Creates a todo list breaking down each feature into specific tasks based on the project architecture*\nAssistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.\n\n<reasoning>\nThe assistant used the todo list because:\n1. The user provided multiple complex features to implement in a comma separated list\n2. The todo list helps organize these large features into manageable tasks\n3. This approach allows for tracking progress across the entire implementation\n</reasoning>\n</example>\n\n<example>\nUser: Can you help optimize my React application It's rendering slowly and has performance issues.</user>\nAssistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.\n*Reviews component structure, render patterns, state management, and data fetching*\nAssistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.\n*Creates todo list with items like: 1) Implement memoization for expensive calculations in ProductList, 2) Add virtualization for long lists in Dashboard, 3) Optimize image loading in Gallery component, 4) Fix state update loops in ShoppingCart, 5) Review bundle size and implement code splitting*\nLet's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant examined the codebase to identify specific performance issues\n2. Based on this analysis, it identified multiple optimization opportunities\n3. Performance optimization is a non-trivial task requiring multiple steps\n4. The todo list helps methodically track improvements across different components\n5. This systematic approach ensures all performance bottlenecks are addressed\n</reasoning>\n</example>\n\n## Examples of When NOT to Use the Todo List\n\n<example>\nUser: How do I print 'Hello World' in Python\nAssistant: In Python, you can print \"Hello World\" with this simple code:\n\npython\nprint(\"Hello World\")\n\nThis will output the text \"Hello World\" to the console when executed.</assistant>\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.\n</reasoning>\n</example>\n\n<example>\nUser: What does the git status command do\nAssistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.\n\n<reasoning>\nThe assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.\n</reasoning>\n</example>\n\n</example>\n\n<example>\nUser: Run npm install for me and tell me what happens.\nAssistant: I'll run the npm install command for you.\n\n*Executes: npm install*\n\nThe command completed successfully. Here's the output:\n[Output of npm install command]\n\nAll dependencies have been installed according to your package.json file.\n\n<reasoning>\nThe assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.\n</reasoning>\n</example>\n\n## Task States and Management\n\n1. **Task States**: Use these states to track progress:\n   - pending: Task not yet started\n   - in_progress: Currently working on (limit to ONE task at a time)\n   - completed: Task finished successfully\n\n2. **Task Management**:\n   - Update task status in real-time as you work\n   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n   - Only have ONE task in_progress at any time\n   - Complete current tasks before starting new ones\n   - Remove tasks that are no longer relevant from the list entirely\n\n3. **Task Completion Requirements**:\n   - ONLY mark a task as completed when you have FULLY accomplished it\n   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n   - When blocked, create a new task describing what needs to be resolved\n   - Never mark a task as completed if:\n     - Tests are failing\n     - Implementation is partial\n     - You encountered unresolved errors\n     - You couldn't find necessary files or dependencies\n\n4. **Task Breakdown**:\n   - Create specific, actionable items\n   - Break complex tasks into smaller, manageable steps\n   - Use clear, descriptive task names\n\n## important\n- Return valid json input\n\n## tool param example\n{\"todos\": [{\"id\": \"1\", \"task\": \"Create the basic HTML structure with a checkerboard grid\", \"status\": \"completed\"}, {\"id\": \"2\", \"task\": \"Add CSS styling for the board, squares, and pieces\", \"status\": \"completed\"}, {\"id\": \"3\", \"task\": \"Implement JavaScript game logic for piece movement and jumps\", \"status\": \"in_progress\"}, {\"id\": \"4\", \"task\": \"Add game state management (player turns, win conditions)\", \"status\": \"pending\"}, {\"id\": \"5\", \"task\": \"Test the game functionality and fix any issues\", \"status\": \"pending\"}]}\n\nWhen in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.\n",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "description": "The updated todo list"
                    }
                },
                "required": [
                    "todos"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_read",
            "description": "Use this tool to read the current to-do list for the session. This tool should be used proactively and frequently to ensure that you are aware of\nthe status of the current task list. You should make use of this tool as often as possible, especially in the following situations:\n- At the beginning of conversations to see what's pending\n- Before starting new tasks to prioritize work\n- When the user asks about previous tasks or plans\n- Whenever you're uncertain about what to do next\n- After completing tasks to update your understanding of remaining work\n- After every few messages to ensure you're on track\n\nUsage:\n- This tool takes in no parameters. So leave the input blank or empty. DO NOT include a dummy object, placeholder string or a key like \"input\" or \"empty\". LEAVE IT BLANK.\n- Returns a list of todo items with their status, priority, and content\n- Use this information to track progress and plan next steps\n- If no todos exist yet, an empty list will be returned",
            "parameters": {
                "type": "object",
                "properties": {},
                "description": "No input is required, leave this field blank. NOTE that we do not require a dummy object, placeholder string or a key like \"input\" or \"empty\". LEAVE IT BLANK."
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "This is a tool for making multiple edits to a single file in one operation. It is built on top of the replace tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the replace tool when you need to make multiple edits to the same file.\n\nBefore using this tool:\n\n1. Use the Read tool to understand the file's contents and context\n2. Verify the directory path is correct\n\nTo make multiple file edits, provide the following:\n1. file_path: The absolute path to the file to modify (must be absolute, not relative)\n2. edits: An array of edit operations to perform, where each edit contains:\n   - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)\n   - new_string: The edited text to replace the old_string\n   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.\n\nIMPORTANT:\n- All edits are applied in sequence, in the order they are provided\n- Each edit operates on the result of the previous edit\n- All edits must be valid for the operation to succeed - if any edit fails, none will be applied\n- This tool is ideal when you need to make several changes to different parts of the same file\n- For Jupyter notebooks (.ipynb files), use the NotebookEdit instead\n\nCRITICAL REQUIREMENTS:\n1. All edits follow the same requirements as the single Edit tool\n2. The edits are atomic - either all succeed or none are applied\n3. Plan your edits carefully to avoid conflicts between sequential operations\n\nWARNING:\n- The tool will fail if edits.old_string doesn't match the file contents exactly (including whitespace)\n- The tool will fail if edits.old_string and edits.new_string are the same\n- Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find\n\nWhen making edits:\n- Ensure all edits result in idiomatic, correct code\n- Do not leave the code in a broken state\n- Always use absolute file paths (starting with /)\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.\n\nIf you want to create a new file, use:\n- A new file path, including dir name if needed\n- First edit: empty old_string and the new file's contents as new_string\n- Subsequent edits: normal edit operations on the created content",
            "parameters": {
                "properties": {
                    "file_path": {
                        "description": "The absolute path to the file to modify"
                    },
                    "edits": {
                        "description": "Array of edit operations to perform sequentially on the file"
                    }
                },
                "required": [
                    "file_path",
                    "edits"
                ],
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Launch a new agent to handle complex, multi-step tasks autonomously. \n\nAvailable agent types and the tools they have access to:\n- general-purpose: For complex research, code searching, and multi-step tasks (Tools: read_file, run_shell_command, glob, search_file_content, list_directory, multi_edit, replace, write_file, todo_read, todo_write, web_fetch, web_search)\n\n\n## CRITICAL: Agent Type Naming Rules\n\n**IMPORTANT**: When using the Task tool, the subagent_type parameter MUST be EXACTLY the same as the agent type name listed above. Do not modify, abbreviate, or change the naming convention.\n\n**Valid agent type names** (copy exactly as written):\n- \"general-purpose\"\n\n\n**Examples of CORRECT usage:**\n- For agent type \"code-reviewer\"  subagent_type: \"code-reviewer\" \n- For agent type \"general-purpose\"  subagent_type: \"general-purpose\"\n- For agent type \"frontend-developer\"  subagent_type: \"frontend-developer\"\n\n**Examples of INCORRECT usage:**\n- code-reviewer  code_review \n- general-purpose  general_purpose  \n- frontend-developer  frontend_developer \n\n## Tool Usage Guidelines\n\nWhen to use the Task tool:\n- When you are instructed to execute custom slash commands. Use the Task tool with the slash command invocation as the entire prompt. The slash command can take arguments. For example: Task(description=\"Check the file\", prompt=\"/check-file path/to/file.py\")\n- For complex, multi-step tasks that match the agent descriptions above\n\nWhen NOT to use the Task tool:\n- If you want to read a specific file path, use the read_file or glob tool instead of the Task tool, to find the match more quickly\n- If you are searching for a specific class definition like \"class Foo\", use the glob tool instead, to find the match more quickly\n- If you are searching for code within a specific file or set of 2-3 files, use the read_file tool instead of the Task tool, to find the match more quickly\n- Other tasks that are not related to the agent descriptions above\n\n## Usage Notes\n\n1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.\n4. The agent's outputs should generally be trusted\n5. Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent\n6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.\n\n## Example Usage\n\n<example>\nuser: \"Please write a function that checks if a number is prime\"\nassistant: Sure let me write a function that checks if a number is prime\nassistant: First let me use the write_file tool to write a function that checks if a number is prime\nassistant: I'm going to use the write_file tool to write the following code:\n<code>\nfunction isPrime(n) {\n  if (n <= 1) return false\n  for (let i = 2; i * i <= n; i++) {\n    if (n % i === 0) return false\n  }\n  return true\n}\n</code>\n<commentary>\nSince a significant piece of code was written and the task was completed, now use the code-reviewer agent to review the code\n</commentary>\nassistant: Now let me use the code-reviewer agent to review the code\nassistant: Uses the task tool with subagent_type: \"code-reviewer\"\n</example>\n\n<example>\nuser: \"Hello\"\n<commentary>\nSince the user is greeting, use the greeting-responder agent to respond with a friendly joke\n</commentary>\nassistant: I'm going to use the task tool with subagent_type: \"greeting-responder\"\n</example>",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "description": "A short (3-5 word) description of the task"
                    },
                    "prompt": {
                        "description": "The task for the agent to perform"
                    },
                    "subagent_type": {
                        "description": "The type of specialized agent to use for this task"
                    }
                },
                "required": [
                    "description",
                    "prompt",
                    "subagent_type"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exit_plan_mode",
            "description": "\nUse this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode. \nIMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.\n\nEg. \n1. Initial task: \"Search for and understand the implementation of vim mode in the codebase\" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.\n2. Initial task: \"Help me implement yank mode for vim\" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.\n",
            "parameters": {
                "properties": {
                    "plan": {
                        "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise."
                    }
                },
                "required": [
                    "plan"
                ],
                "type": "object"
            }
        }
    }
]

request_data = {
    "model": "glm-4.5-air",
    "messages": messages,
    "max_tokens": 2048,
    "temperature": 1.0,
    "tools": tools,
    "stream": False,
}
response = requests.post(
    # f"http://127.0.0.1:8000/v1/chat/completions",
    f"https://open.bigmodel.cn/api/paas/v4/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer e2c883fb4bc243e28f62c99c85d89c01.9MVM2f666ywn803l"
    },
    json=request_data,
    timeout=360
)

print(response.json())
