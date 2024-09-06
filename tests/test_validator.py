import os
from validator import BespokeMiniCheck
os.environ["BESPOKE_API_KEY"] = "bespoke-demo-key"
bespoke_minicheck_validator = BespokeMiniCheck(on_fail="fix")


def test_pass():
    test_output = "Alex likes dogs."
    result = bespoke_minicheck_validator.validate(test_output, metadata={"context": "Alex likes dogs, but not cats."})

    assert result.outcome == "pass"

def test_fail():
    test_output = "Alex likes cats. Alex likes dogs."
    result = bespoke_minicheck_validator.validate(test_output, metadata={"context": "Alex likes dogs, but not cats."})

    assert result.outcome == "fail"
    assert result.fix_value == "Alex likes dogs."
