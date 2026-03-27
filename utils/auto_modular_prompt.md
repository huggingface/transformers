You are an expert contributor to the HuggingFace Transformers library. Your task is to write a modular_{model_name}.py file following the library's modular architecture pattern: inherit from the closest matching existing model and only override what genuinely differs. Output ONLY valid Python source code — no markdown fences, no explanation.

{prompt}

---

Full source of `{modeling_file_name}` (the model being integrated):

```python
{modeling_code}
```

---

Reference modular file (`modular_gemma.py`) showing the expected style and structure:

```python
{ref_code}
```

---

Now write the complete `modular_{model_name}.py`. Output only the Python source code.
