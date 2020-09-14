import fire
try:
    from .run_eval import generate_summaries_or_translations
except ImportError:
    from run_eval import generate_summaries_or_translations
