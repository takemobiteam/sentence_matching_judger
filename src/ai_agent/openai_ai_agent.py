import json
from copy import deepcopy
from openai import OpenAI


def get_n_characters_from_msgs(msgs: list) -> int:
    """get number of characters in msgs to api based llms

    Args:
        msgs (list): list of conversation records

    Returns:
        int: total number of input tokens for a llm api key
    """
    total_input_len = 0
    for msg in msgs:
        content = msg["content"]
        total_input_len += len(content)
    return total_input_len


def get_subquestion_res(
    msgs: list,
    new_msg: str,
    client: OpenAI,
    prompts: dict,
    n_input_chars: int,
    n_output_chars: int,
):
    """run llm to get result of a subquestion

    Args:
        msgs (list): chat records
        new_msg (str): new content from user
        client (openai.api_resources.chat_completion.ChatCompletion): openai class for chat completion
        prompts (dict): prompt for extracting answer
        n_input_chars (int): total number of input characters
        n_output_chars (int): total number of output characters

    Returns:
        new_msgs (list): new dialogue records
        answer (str): yes no or idk answer for the subquestion
        n_input_chars (int): total number of characters as input to the llm up to now
        n_output_chars (int):total number of characters as output from the llm up to now
    """

    # ask the subquestion and get response
    new_msgs = deepcopy(msgs) + [{"role": "user", "content": new_msg}]
    n_input_chars += get_n_characters_from_msgs(new_msgs)
    response_str = (
        client.chat.completions.create(model="gpt-4o-mini", messages=new_msgs)
        .choices[0]
        .message.content
    )
    n_output_chars += len(response_str)
    new_msgs += [{"role": "assistant", "content": response_str}]

    # extract the final answer from the response that has both reasoning and answer.
    extract_answer_msg = [
        {"role": "user", "content": prompts["extract_answer"] + "\n" + response_str}
    ]
    n_input_chars += get_n_characters_from_msgs(extract_answer_msg)
    answer = (
        client.chat.completions.create(model="gpt-4o-mini", messages=extract_answer_msg)
        .choices[0]
        .message.content
    )
    n_output_chars += len(answer)
    return new_msgs, answer, n_input_chars, n_output_chars


def ai_agent(data_point: dict, 
             tag: dict, 
             client: OpenAI, 
             prompts: dict, 
             matched_line_only: bool):
    """main ai agent code,
       if checks the tag assignment from 4 perspectives first
       1. wrong meaning: at the beginning it's only for partial matching, for example 
          "accessible pool" and "seasonal accessible pool". But it's expanded to mean more,
          unfortunately now wrong meaning become a pocket error class. Almost anything can
          be wrong meaning. But the overall accy now is okay and separate different cause is
          not the main task. So I will only may be improve it in the future.
        2 nearby: the tag describe something near the poi but not the poi itself
        3. negative: things like not limited hot water, no pizza, no microwave
        4. low quality:: we don't want to assign a tag when user's attitude toward the tag is negative
            example: broken outdoor furniture, pizza tastes awful
        5. outdated: This location was a jazz bar 20 years ago.
        6. other: other
        More detailed definition in prompts

        if any mistake is found the program halts and return,
        otherwise the llm will double check if the assignment is correct and output final result.  

    Args:
        data_point (dict): a dict with selected information for the poi
        tag (dict): tag represented as a dict, with keys parent and name
                    ex: {parent: architecture style, name: A Frame}
        client (OpenAI): open ai api
        prompts (dict): a dictionay that contains multiple prompts for the llm
        matched_line_only (bool): set to true if we only give llm the name of the poi and the matched sentence
                                  set to False if we give llm name, description reviews of the poi and the matched sentence

    Returns:
        ret_msgs (list): selected assistant answers
        causes (list): list of casuese for the kind of error, causes are (wrong_meaning, negative, low qualilty, outdated and nearby) 
        n_input_chars (int): number of input characters
        answer (str): final answer for if the tag assign is not valid, 
                      "yes" if the tag assignment is not valid, "no" if the tag assignment is valid
        n_output_chars (int): number of output characters.
    """
    if matched_line_only:
        data_point = {
            "name": data_point["name"],
            "type": data_point["type"],
            "matched_line": data_point["matched_line"],
            "matched_field": data_point["matched_field"],
        }
    curr_messages = [{"role": "system", "content": prompts["system_prompt"]}] + [
        {
            "role": "user",
            "content": f"data point: {json.dumps(data_point, ensure_ascii=False, indent=4)} \n tag: {json.dumps(tag, ensure_ascii=False, indent=4)}",
        }
    ]
    curr_messages += [{"role": "assistant", "content": "What is your question?"}]

    causes = []
    n_input_chars, n_output_chars = 0, 0
    part_of_a_phrase_msgs, part_of_a_phrase_answer, n_input_chars, n_output_chars = (
        get_subquestion_res(
            curr_messages,
            prompts["wrong_meaning"],
            client,
            prompts,
            n_input_chars,
            n_output_chars,
        )
    )
    phrase_start_ind = -2
    if part_of_a_phrase_answer == "yes":
        causes.append("wrong_meaning")

    nearby_msgs, nearby_answer, n_input_chars, n_output_chars = get_subquestion_res(
        curr_messages, prompts["nearby"], client, prompts, n_input_chars, n_output_chars
    )
    if nearby_answer == "yes":
        causes.append("nearby")

    negative_msgs, negative_answer, n_input_chars, n_output_chars = get_subquestion_res(
        curr_messages,
        prompts["negative"],
        client,
        prompts,
        n_input_chars,
        n_output_chars,
    )
    if negative_answer == "yes":
        causes.append("negative")

    outdated_msgs, outdated_answer, n_input_chars, n_output_chars = get_subquestion_res(
        curr_messages,
        prompts["outdated_relevancy"],
        client,
        prompts,
        n_input_chars,
        n_output_chars,
    )
    if outdated_answer == "yes":
        causes.append("outdated")

    badq_msgs, badq_answer, n_input_chars, n_output_chars = get_subquestion_res(
        curr_messages,
        prompts["bad_quality"],
        client,
        prompts,
        n_input_chars,
        n_output_chars,
    )
    if badq_answer == "yes":
        causes.append("bad_quality")

    if not len(causes):
        final_check_msg = (
            curr_messages
            + part_of_a_phrase_msgs[phrase_start_ind:]
            + nearby_msgs[-2:]
            + negative_msgs[-2:]
            + outdated_msgs[-2:]
            + badq_msgs[-2:]
        )
        final_msgs, final_answer, n_input_chars, n_output_chars = get_subquestion_res(
            final_check_msg,
            prompts["other"],
            client,
            prompts,
            n_input_chars,
            n_output_chars,
        )
        if final_answer == "yes":
            return final_msgs, ["other"], final_answer, n_input_chars, n_output_chars
        else:
            return final_msgs, [], final_answer, n_input_chars, n_output_chars
    else:
        ret_msgs = curr_messages
        for cause in causes:
            if cause == "wrong_meaning":
                ret_msgs += part_of_a_phrase_msgs[phrase_start_ind:]
            if cause == "nearby":
                ret_msgs += nearby_msgs[-2:]
            if cause == "negative":
                ret_msgs += negative_msgs[-2:]
            if cause == "outdated":
                ret_msgs += outdated_msgs[-2:]
            if cause == "bad_quality":
                ret_msgs += badq_msgs[-2:]
        return ret_msgs, causes, "yes", n_input_chars, n_output_chars
