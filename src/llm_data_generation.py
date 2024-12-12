from openai import OpenAI
import json
import pandas as pd
import os
from tqdm import tqdm

from src.string_matching.string_match import find_match
from src.ai_agent.load_prompts import load_prompts
from src.ai_agent.openai_ai_agent import ai_agent


def estimate_price(n_input_chars: int, n_output_chars: int) -> float:
    """estimate the price per sample

    Args:
        n_input_chars (int): number of input characters
        n_output_chars (int): number of output characters

    Returns:
        float: estimated price per sample
    """
    avg_n_input_chars = sum(n_input_chars) / len(n_input_chars)
    avg_n_output_chars = sum(n_output_chars) / len(n_output_chars)
    input_token_pricing = 0.15 / 1000000
    output_token_pricing = 0.6 / 1000000
    estimated_token_char_ratio = 0.25
    estimated_price = (
        avg_n_input_chars * estimated_token_char_ratio * input_token_pricing
        + avg_n_output_chars * estimated_token_char_ratio * output_token_pricing
    )
    print(f"estimated price is {estimated_price} per sample")


def api_llm_based_sentence_judge_agent(
    testing_type: str,
    sampled_string_matching_result: pd.DataFrame,
    curr_type_taxonomy: pd.DataFrame,
    matched_line_only: bool,
    openai_api_key: str,
    config: dict,
):
    """Use an ai agent to judge if tags assigned by a string matching
       tag assignment algorithm is valid
       result is saved to a dataframe

    Args:
        testing_type (str): type of the poi, can be experience, dining or lodging
        sampled_string_matching_result (pd.DataFrame): data frame for string matching result
        curr_type_taxonomy (pd.DataFrame): current taxonomy for the specific type of poi
        matched_line_only (bool): set to true if you don't want to input description and reviews to llm
        openai_api_key (str): openai api key
        config (dict): configuration as a dictionary
    """

    # load data and preprocess
    # the goal of the preprocess here is to unify the parent for each tag, For now I use all tags that
    # share the same tag name, ignoring the parent. But I want to use the parent information in llm calls,
    # so I give every sample same parent, for example in the data there's "property feature_beach" and "on site feature_beach",
    # I'll use samples with both tags but unify the parent to on site feature if on site feature is the parent
    # appeared in taxonomy
    curr_type_taxonomy = pd.read_csv(curr_type_taxonomy)
    curr_type_taxonomy = curr_type_taxonomy[["Tag Parent", "Tag Name"]]
    curr_type_taxonomy = curr_type_taxonomy.rename(
        columns={"Tag Parent": "real_parent", "Tag Name": "tag_name"}
    )
    tag_data = pd.read_csv(sampled_string_matching_result)
    tag_data = pd.merge(tag_data, curr_type_taxonomy, on="tag_name", how="left")
    tag_data = tag_data.drop_duplicates(subset=["name", "source", "mobi_id"])
    if config["run_test_mode"]:
        tag_data = tag_data[
            tag_data["tag_name"].isin(config[f"{testing_type}_test_keys"])
        ]

    prompts = load_prompts()
    res_list = []
    n_input_chars = []
    n_output_chars = []
    for _, row in tqdm(tag_data.iterrows(), total=len(tag_data)):
        curr_sample_res = {}
        data_point = {
            "name": row["name"],
            "description": row["description"],
            "type": row["type"],
            "review": row["review0"],
        }

        matched_field_name, matched_line = find_match(row["tag_name"], row)

        if not matched_line:
            # I only kept 3 reviews, so it's possible that the matched part is in reviews I didn't include,
            # For such case I'll skip
            curr_sample_res = {
                "name": row["name"],
                "matched_line": None,
                "tag_parent": row["real_parent"],
                "tag_name": row["tag_name"],
                "has_conflict": None,
                "causes": [],
            }
        else:
            data_point["matched_field_name"] = matched_field_name
            data_point["matched_line"] = matched_line
            matched_field = row[matched_field_name]
            data_point["matched_field"] = matched_field
            tag = {"parent": row["real_parent"], "name": row["tag_name"]}

            client = OpenAI(api_key=openai_api_key)
            llm_res, causes, answer, sample_n_input_chars, sample_n_output_chars = (
                ai_agent(
                    data_point,
                    tag,
                    client,
                    prompts,
                    matched_line_only=matched_line_only,
                )
            )
            n_input_chars.append(sample_n_input_chars)
            n_output_chars.append(sample_n_output_chars)
            llm_res = [rec for rec in llm_res if rec["role"] == "assistant"]
            curr_sample_res = {
                "name": row["name"],
                "matched_line": data_point["matched_line"],
                "tag_parent": row["real_parent"],
                "tag_name": row["tag_name"],
                "eval_res": json.dumps(llm_res),
                "has_conflict": answer,
                "causes": causes,
            }
        res_list.append(curr_sample_res)
    res = pd.DataFrame(res_list)
    os.makedirs("result", exist_ok=True)
    estimate_price(n_input_chars, n_output_chars)
    if matched_line_only:
        res.to_csv(
            f"result/{testing_type}_tag_eval_res_matched_line_only.csv", index=False
        )
    else:
        res.to_csv(f"result/{testing_type}_tag_eval_res.csv", index=False)
