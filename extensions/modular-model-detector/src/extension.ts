import * as vscode from "vscode";
import * as path from "path";
import * as os from "os";
import * as fs from "fs/promises";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

type DetectorMatch = {
  identifier: string;
  relative_path: string;
  match_name: string;
  class_name: string | null;
  method_name: string | null;
  score: number;
  release_date: string;
  full_path: string;
  line: number | null;
};

type DetectorResult = {
  kind: string;
  class_name: string | null;
  method_name: string | null;
  embedding: DetectorMatch[];
  jaccard?: DetectorMatch[];
  intersection?: string[];
};

type DetectorPayload = {
  modeling_file: string;
  precision: string;
  granularity: string;
  release_date: string;
  best_candidate: {
    relative_path: string;
    full_path: string;
    release_date: string;
    score: number;
  } | null;
  results: Record<string, DetectorResult>;
};

type AiSuggestion = {
  title: string;
  detail?: string;
  target_full_path?: string;
  line?: number;
};

type AiPayload = {
  suggestions?: Record<string, AiSuggestion[]>;
};

type CachedAnalysis = {
  payload: DetectorPayload;
  suggestions: Record<string, AiSuggestion[]>;
};

const analysisCache = new Map<string, CachedAnalysis>();

function getWorkspaceRoot(document: vscode.TextDocument): string | null {
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
  if (!workspaceFolder) {
    return null;
  }
  return workspaceFolder.uri.fsPath;
}

async function ensureFileOnDisk(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function findQueryName(
  symbols: vscode.DocumentSymbol[],
  position: vscode.Position,
  parentClassName: string | null = null
): string | null {
  for (const symbol of symbols) {
    if (!symbol.range.contains(position)) {
      continue;
    }
    const nextParentClass =
      symbol.kind === vscode.SymbolKind.Class ? symbol.name : parentClassName;
    const childMatch = findQueryName(symbol.children, position, nextParentClass);
    if (childMatch) {
      return childMatch;
    }
    if (
      symbol.kind === vscode.SymbolKind.Function ||
      symbol.kind === vscode.SymbolKind.Method
    ) {
      return nextParentClass ? `${nextParentClass}.${symbol.name}` : symbol.name;
    }
    if (symbol.kind === vscode.SymbolKind.Class) {
      return symbol.name;
    }
    if (parentClassName) {
      return parentClassName;
    }
  }
  return null;
}

async function getQueryNameAtPosition(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<string | null> {
  const symbols = (await vscode.commands.executeCommand(
    "vscode.executeDocumentSymbolProvider",
    document.uri
  )) as vscode.DocumentSymbol[] | undefined;
  if (!symbols || symbols.length === 0) {
    return null;
  }
  return findQueryName(symbols, position);
}

async function runDetector(
  scriptPath: string,
  pythonPath: string,
  precision: string,
  granularity: string,
  useJaccard: boolean,
  topK: number,
  modelingFile: string,
  cwd: string
): Promise<{ payload: DetectorPayload; outputPath: string }> {
  const outputPath = path.join(os.tmpdir(), `modular-model-detector-${Date.now()}.json`);
  const args = [
    scriptPath,
    "--modeling-file",
    modelingFile,
    "--precision",
    precision,
    "--granularity",
    granularity,
    "--top-k",
    String(topK),
    "--output-json",
    outputPath,
  ];
  if (useJaccard) {
    args.push("--use_jaccard", "True");
  }
  await execFileAsync(pythonPath, args, { cwd });
  const raw = await fs.readFile(outputPath, "utf-8");
  const payload = JSON.parse(raw) as DetectorPayload;
  return { payload, outputPath };
}

async function runAiPostProcessor(
  command: string,
  args: string[],
  inputPath: string,
  cwd: string
): Promise<AiPayload | null> {
  const outputPath = path.join(os.tmpdir(), `modular-model-ai-${Date.now()}.json`);
  const finalArgs = [...args, inputPath, outputPath];
  await execFileAsync(command, finalArgs, { cwd });
  const raw = await fs.readFile(outputPath, "utf-8");
  await fs.unlink(outputPath);
  return JSON.parse(raw) as AiPayload;
}

async function openMatch(match: DetectorMatch): Promise<void> {
  const targetUri = vscode.Uri.file(match.full_path);
  const doc = await vscode.workspace.openTextDocument(targetUri);
  const editor = await vscode.window.showTextDocument(doc, { preview: false });
  if (match.line) {
    const line = Math.max(match.line - 1, 0);
    const range = new vscode.Range(line, 0, line, 0);
    editor.selection = new vscode.Selection(range.start, range.start);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    const decoration = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      backgroundColor: "rgba(255, 208, 0, 0.25)",
    });
    editor.setDecorations(decoration, [range]);
    setTimeout(() => decoration.dispose(), 2500);
  }
}

async function showSuggestion(suggestion: AiSuggestion): Promise<void> {
  if (suggestion.target_full_path) {
    await openMatch({
      identifier: "",
      relative_path: "",
      match_name: suggestion.title,
      class_name: null,
      method_name: null,
      score: 0,
      release_date: "",
      full_path: suggestion.target_full_path,
      line: suggestion.line ?? null,
    });
  }
  if (suggestion.detail) {
    vscode.window.showInformationMessage(suggestion.detail);
  } else {
    vscode.window.showInformationMessage(suggestion.title);
  }
}

class ModularModelCodeActionProvider implements vscode.CodeActionProvider {
  async provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
    _context: vscode.CodeActionContext,
    _token: vscode.CancellationToken
  ): Promise<vscode.CodeAction[]> {
    const actions: vscode.CodeAction[] = [];
    const docKey = document.uri.toString();
    const cached = analysisCache.get(docKey);
    if (!cached) {
      const action = new vscode.CodeAction(
        "Run modular detector on this file",
        vscode.CodeActionKind.QuickFix
      );
      action.command = {
        command: "modularModelDetector.analyzeFile",
        title: "Run modular detector on this file",
      };
      actions.push(action);
      return actions;
    }

    const queryName = await getQueryNameAtPosition(document, range.start);
    if (!queryName) {
      return actions;
    }
    const result = cached.payload.results[queryName];
    if (!result) {
      return actions;
    }

    for (const match of result.embedding) {
      const action = new vscode.CodeAction(
        `Open match: ${match.match_name} (${match.score.toFixed(4)})`,
        vscode.CodeActionKind.QuickFix
      );
      action.command = {
        command: "modularModelDetector.openMatch",
        title: "Open match",
        arguments: [match],
      };
      actions.push(action);
    }

    const suggestionList = cached.suggestions[queryName] ?? [];
    for (const suggestion of suggestionList) {
      const action = new vscode.CodeAction(
        `AI: ${suggestion.title}`,
        vscode.CodeActionKind.QuickFix
      );
      action.command = {
        command: "modularModelDetector.showSuggestion",
        title: "Show suggestion",
        arguments: [suggestion],
      };
      actions.push(action);
    }

    return actions;
  }
}

export function activate(context: vscode.ExtensionContext) {
  const analyzeDisposable = vscode.commands.registerCommand(
    "modularModelDetector.analyzeFile",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor.");
        return;
      }
      const workspaceRoot = getWorkspaceRoot(editor.document);
      if (!workspaceRoot) {
        vscode.window.showErrorMessage("Open the file inside a workspace.");
        return;
      }
      const activePath = editor.document.uri.fsPath;
      const fileExists = await ensureFileOnDisk(activePath);
      if (!fileExists) {
        vscode.window.showErrorMessage(
          `Active file is not on disk: ${activePath}. Save the file and try again.`
        );
        return;
      }

      const config = vscode.workspace.getConfiguration("modularModelDetector");
      const pythonPath = config.get<string>("pythonPath", "python");
      const precision = config.get<string>("precision", "float32");
      const granularity = config.get<string>("granularity", "method");
      const topK = config.get<number>("topK", 5);
      const useJaccard = config.get<boolean>("useJaccard", false);
      const scriptOverride = config.get<string>("scriptPath", "").trim();
      const scriptPath =
        scriptOverride || path.join(workspaceRoot, "utils", "modular_model_detector.py");
      const aiCommand = config.get<string>("aiCommand", "").trim();
      const aiArgs = config.get<string[]>("aiArgs", []);

      const modelingFile = activePath.startsWith(workspaceRoot + path.sep)
        ? path.relative(workspaceRoot, activePath)
        : activePath;

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Running modular model detector",
          cancellable: false,
        },
        async () => {
          let outputPath = "";
          try {
            const result = await runDetector(
              scriptPath,
              pythonPath,
              precision,
              granularity,
              useJaccard,
              topK,
              modelingFile,
              workspaceRoot
            );
            outputPath = result.outputPath;
            const suggestions: Record<string, AiSuggestion[]> = {};
            if (aiCommand) {
              try {
                const aiPayload = await runAiPostProcessor(
                  aiCommand,
                  aiArgs,
                  outputPath,
                  workspaceRoot
                );
                if (aiPayload?.suggestions) {
                  Object.assign(suggestions, aiPayload.suggestions);
                }
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                vscode.window.showErrorMessage(`AI post-processor failed: ${message}`);
              }
            }
            analysisCache.set(editor.document.uri.toString(), {
              payload: result.payload,
              suggestions,
            });
            vscode.window.showInformationMessage(
              `Detector finished: ${Object.keys(result.payload.results).length} symbols.`
            );
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Detector failed: ${message}`);
          } finally {
            if (outputPath) {
              try {
                await fs.unlink(outputPath);
              } catch {
                // ignore cleanup errors
              }
            }
          }
        }
      );
    }
  );

  const openMatchDisposable = vscode.commands.registerCommand(
    "modularModelDetector.openMatch",
    async (match: DetectorMatch) => {
      await openMatch(match);
    }
  );

  const showSuggestionDisposable = vscode.commands.registerCommand(
    "modularModelDetector.showSuggestion",
    async (suggestion: AiSuggestion) => {
      await showSuggestion(suggestion);
    }
  );

  const codeActions = vscode.languages.registerCodeActionsProvider(
    { language: "python", scheme: "file" },
    new ModularModelCodeActionProvider(),
    { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
  );

  context.subscriptions.push(
    analyzeDisposable,
    openMatchDisposable,
    showSuggestionDisposable,
    codeActions
  );
}

export function deactivate() {}
