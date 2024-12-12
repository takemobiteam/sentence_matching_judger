def load_prompts():
    """load prompts"""
    out_dict = {}
    with open("prompts/ai-agent-prompts/system_prompt.txt", "r") as f:
        out_dict["system_prompt"] = f.read()

    with open("prompts/ai-agent-prompts/extract_answer.txt", "r") as f:
        out_dict["extract_answer"] = f.read()

    with open("prompts/ai-agent-prompts/wrong_meaning.txt", "r") as f:
        out_dict["wrong_meaning"] = f.read()

    with open("prompts/ai-agent-prompts/nearby.txt", "r") as f:
        out_dict["nearby"] = f.read()

    with open("prompts/ai-agent-prompts/negative.txt", "r") as f:
        out_dict["negative"] = f.read()

    with open("prompts/ai-agent-prompts/outdated_relevancy.txt", "r") as f:
        out_dict["outdated_relevancy"] = f.read()

    with open("prompts/ai-agent-prompts/other.txt", "r") as f:
        out_dict["other"] = f.read()

    with open("prompts/ai-agent-prompts/bad_quality.txt", "r") as f:
        out_dict["bad_quality"] = f.read()

    return out_dict
