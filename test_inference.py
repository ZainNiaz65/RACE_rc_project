from src.inference import verify_answer

def test_dummy():
    article = "Paris is the capital of France."
    question = "What is the capital of France?"
    options = {
        "A": "Berlin",
        "B": "Madrid",
        "C": "Paris",
        "D": "Rome"
    }

    pred, score = verify_answer(article, question, options)
    assert pred in ["A","B","C","D"]