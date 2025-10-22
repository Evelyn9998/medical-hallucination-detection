def physiological_plausibility_check(question, answer):
    rules = [
        ("heart rate", "increase", False),
        ("heart rate", "decrease", True),
        ("oxygen", "decrease", True),
        ("blood pressure", "increase", False),
        ("temperature", "decrease", True)
    ]

    for key, keyword, implausible in rules:
        if key in question.lower() and keyword in answer.lower() and implausible:
            return {
                "physiology_flag": True,
                "note": f"Answer may be physiologically implausible ({key} shouldn't {keyword})."
            }
    return {"physiology_flag": False, "note": "Physiologically plausible."}
