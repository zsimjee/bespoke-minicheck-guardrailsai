from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import os
import requests
import nltk
import concurrent.futures


nltk.download('punkt', quiet=True)

@register_validator(name="guardrails/bespokeai_factcheck", data_type="string")
class BespokeAIFactCheck(Validator):
    # If you don't have any init args, you can omit the __init__ method.
    def __init__(
        self,
        threshold: float = 0.5,
        split_sentences: bool = True,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self._split_sentences = split_sentences

    def is_supported_claim(self, contexts: list[str], claim: str, threshold: float) -> bool:
        key = os.environ["BESPOKEAI_API_KEY"]
        output = requests.post(
            "https://api.bespokelabs.ai/v0/argus/factcheck",
            json={
                "contexts": contexts,
                "claim": claim,
            },
            headers={"api_key": key},
        ).json()

        return any([score >= threshold for score in output["claim_supported_by_contexts"]])

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        threshold = self._threshold if "threshold" not in metadata else metadata["threshold"]
        split_sentences = self._split_sentences if "split_sentences" not in metadata else metadata["split_sentences"]
        contexts = metadata.get("contexts", [])

        if not (len(contexts) > 0 and isinstance(contexts, list)):
            raise ValueError("contexts must be a list of strings")

        if split_sentences:
            claims = nltk.sent_tokenize(value)
        else:
            claims = [value]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.is_supported_claim, contexts, claim, threshold): claim for claim in claims}
            claim_to_supported = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}

        supported = all(claim_to_supported.values())

        if supported:
            return PassResult()

        return FailResult(
            error_message="Claim not supported by BespokeAI",
            fix_value=" ".join([claim for claim, supported in claim_to_supported.items() if supported]),
        )
