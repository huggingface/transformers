from .base import SafetyChecker, SafetyConfig, SafetyResult

# A tiny default checker useful for tests and as an example. Real checkers should subclass SafetyChecker.
class DummySafetyChecker(SafetyChecker):
    """A very small example SafetyChecker used for tests and documentation.

    It marks any text containing the substring "badword" as unsafe and returns a single
    forbidden token id to illustrate how logits suppression works.
    """
    def __init__(self, forbidden_token_id: int = 0):
        self.forbidden_token_id = forbidden_token_id

    def check_texts(self, texts):
        results = []
        for t in texts:
            is_safe = "badword" not in t
            score = 1.0 if is_safe else 0.0
            forbidden = [] if is_safe else [self.forbidden_token_id]
            results.append(SafetyResult(is_safe=is_safe, score=score, forbidden_token_ids=forbidden))
        return results

__all__ = ["SafetyChecker", "SafetyConfig", "SafetyResult", "DummySafetyChecker"]
