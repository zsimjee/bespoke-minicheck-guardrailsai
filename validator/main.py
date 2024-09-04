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
    """Validates that the LLM-generated text is supported by the provided
    context using BespokeAI's factcheck API.

    This validator uses BespokeAI's factcheck API to evaluate the generated text
    against the provided context.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `guardrails/bespokeai_factcheck`    |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Returns supported claims            |

    Args:
        threshold (float, optional): The minimum score for a claim to be
            considered supported. Defaults to 0.5.
        split_sentences (bool, optional): Whether to split the input into
            sentences for individual evaluation. Defaults to True.
        on_fail (Optional[Callable], optional): A callable to execute when the
            validation fails. Defaults to None.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        split_sentences: bool = True,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self._split_sentences = split_sentences

    def _chunking_function(self, chunk: str) -> list[str]:
        return nltk.sent_tokenize(chunk)

    def _inference_local(self, model_input: Any):
        raise NotImplementedError("Local inference is not supported for BespokeAIFactCheck validator.")

    def _inference_remote(self, model_input: Any):
        key = os.environ["BESPOKEAI_API_KEY"]
        output = requests.post(
            "https://api.bespokelabs.ai/v0/argus/factcheck",
            json=model_input,
            headers={"api_key": key},
        ).json()
        return output

    def _validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
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
            futures = [executor.submit(self._inference, {"claim": claim, "contexts": contexts}) for claim in claims]
            is_supported = [future.result() for future in concurrent.futures.as_completed(futures)]

        all_claims_supported = all(
            any(score >= threshold for score in scores['claim_supported_by_contexts'])
            for scores in is_supported
        )

        if all_claims_supported:
            return PassResult()
        else:
            supported_claims = [
                claim
                for claim, supported in zip(claims, is_supported)
                if any(score >= threshold for score in supported['claim_supported_by_contexts'])
            ]

            return FailResult(
                error_message="Claim not supported by BespokeAI",
                fix_value=" ".join(supported_claims),
            )
