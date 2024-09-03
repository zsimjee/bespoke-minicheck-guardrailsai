from validator import BespokeAIFactCheck

validator = BespokeAIFactCheck(on_fail="fix")


def test_pass():
    test_output = "Alex likes dogs."
    result = validator.validate(test_output, metadata={"contexts": ["Alex likes dogs, but not cats."]})

    assert result.outcome == "pass"

def test_fail():
    test_output = "Alex likes cats. Alex likes dogs."
    result = validator.validate(test_output, metadata={"contexts": ["Alex likes dogs, but not cats."]})

    print(result)

    assert result.outcome == "fail"
    assert result.fix_value == "Alex likes dogs."
