# Modular Model Detector (VSCode/Cursor)

Run `utils/modular_model_detector.py` from VSCode/Cursor, then use contextual actions to jump to similar model components.

## Usage

- Open a Python modeling file.
- Run `Transformers: Analyze File for Modular Suggestions` from the command palette or editor context menu.
- Use the lightbulb or right-click actions inside a method to open similar matches.

## Settings

- `modularModelDetector.pythonPath`: Python executable to run the detector.
- `modularModelDetector.scriptPath`: Override script path (defaults to `utils/modular_model_detector.py` in the workspace).
- `modularModelDetector.precision`: Embedding precision (`float32`, `int8`, `binary`).
- `modularModelDetector.granularity`: Index granularity (`definition`, `method`).
- `modularModelDetector.topK`: Number of matches per symbol.
- `modularModelDetector.useJaccard`: Include Jaccard results in detector output (currently only embedding matches are shown).
- `modularModelDetector.aiCommand`: Optional command for AI post-processing (input JSON path + output JSON path).
- `modularModelDetector.aiArgs`: Extra args passed to the AI command (before input/output paths).

## AI Command Contract

If `aiCommand` is set, the extension will call:

```
aiCommand <aiArgs...> <input_json> <output_json>
```

The output JSON should look like:

```json
{
  "suggestions": {
    "ClassName.method_name": [
      {
        "title": "Replace with ExistingMethod",
        "detail": "Short rationale for the change",
        "target_full_path": "/abs/path/to/file.py",
        "line": 123
      }
    ]
  }
}
```
