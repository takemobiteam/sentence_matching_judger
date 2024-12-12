"""
Given sampled point of interests that contains tag that's assigned 
based on string matching algorithm, use an expensive llm agent
to judge if the string matching algorithm assigns the tag correctly.

The judging result and may be the reasoning will be used for training
a smaller llm model for the same task.
"""

import argparse
import yaml
from src.llm_data_generation import api_llm_based_sentence_judge_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testing_type", type=str, default="lodging")
    args = parser.parse_args()

    with open("configs/default_config.yml", "r") as f:
        config = yaml.safe_load(f)

    testing_type = args.testing_type
    matched_line_only = config["matched_line_only"]
    sampled_string_matching_result = f"data/{testing_type}_tag_eval.csv"
    curr_type_taxonomy = f"data/Mobi Taxonomy v2 - {testing_type} Taxonomy.csv"
    api_llm_based_sentence_judge_agent(
        testing_type,
        sampled_string_matching_result,
        curr_type_taxonomy,
        matched_line_only,
        config["api_key"],
        config,
    )
