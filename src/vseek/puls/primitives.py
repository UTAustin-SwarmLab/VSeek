from vseek.puls.llm import *
from vseek.puls.prompts import *
import json
import os
import re
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.videomme import VideoMME
from data.mlvu import MLVU
from data.cgbench import CGBench
from concurrent.futures import ThreadPoolExecutor

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def process_specification(specification, propositions):
    new_propositions = []
    for prop in propositions:
        prop_cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", prop)
        prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
        prop_cleaned = prop_cleaned.replace("'", "").replace("-", "_").lower()
        prop_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', prop_cleaned)
        new_propositions.append(prop_cleaned)

    replacements = sorted(
        list(zip(propositions, new_propositions)),
        key=lambda x: len(x[0]),
        reverse=True
    )
    for original, new in replacements:
        if specification.count(original) == 1:
            specification = specification.replace(original, f'"{new}"')

    replacements = {
        "AND": "&",
        "OR": "|",
        "UNTIL": "U",
        "ALWAYS": "G",
        "EVENTUALLY": "F",
        "NOT": "!"
    }
    for word, symbol in replacements.items():
        specification = specification.replace(word, symbol)

    # specification = specification.replace("U", "& F")
    # if 'G "' in specification:
    #     specification = specification.replace('G "', 'F "')

    return new_propositions, specification

def resolve_correct_answer_text(candidates, correct_choice):
    if not isinstance(candidates, list) or not candidates:
        return ""

    if isinstance(correct_choice, int):
        idx = correct_choice
    else:
        choice_str = str(correct_choice).strip()
        if choice_str.isdigit():
            idx = int(choice_str)
        else:
            letter_match = re.fullmatch(r"[A-Za-z]", choice_str)
            if letter_match:
                idx = ord(choice_str.upper()) - ord("A")
            else:
                return ""

    if 0 <= idx < len(candidates):
        return str(candidates[idx])
    return ""

def PULS(llm, prompt, openai_key=None):

    full_prompt = find_prompt(prompt)
    llm_output = llm.prompt(full_prompt)
    print("LLM Output: ", llm_output)
    parsed = clean_and_parse_json(llm_output)

    final_output = {}

    cleaned_props, processed_spec = process_specification(parsed["specification"], parsed["proposition"])
    final_output["proposition"] = cleaned_props
    final_output["specification"] = processed_spec
    return final_output


import argparse


@hydra.main(version_base=None, config_path="../../../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
     
    if cfg.dataset.name == "lvb":
        dataset = LongVideoBench(cfg)
    elif cfg.dataset.name == "lvbench":
        dataset = LVBench(cfg)
    elif cfg.dataset.name == "videomme":
        dataset = VideoMME(cfg)
    elif cfg.dataset.name == "mlvu":
        dataset = MLVU(cfg)      
    elif cfg.dataset.name == "cgbench":
        dataset = CGBench(cfg)

    openai_key = os.getenv("OPENAI_API_KEY")


    def _process_entry(entry):
        llm = LLM(openai_api_key=openai_key)

        prompt = entry["question"].split("Question:")[-1].strip()
        candidates = entry.get("candidates", [])
        correct_choice = entry.get("correct_choice")
        correct_answer_text = resolve_correct_answer_text(candidates, correct_choice)
        if correct_answer_text:
            prompt = prompt + "\n Correct Answer: " + correct_answer_text
        else:
            prompt = prompt + "\n Correct Answer Index (0-based): " + str(correct_choice)
        print("Question: ", prompt)
        try:    
            output = PULS(llm, prompt)
        except Exception as e:
            output = {
                "proposition": [],
                "specification": ""
            }
        print(output)
        entry["puls"] = output
        return entry

    entries = dataset.load_data()
    max_workers = int(os.getenv("VSEEK_WORKERS", 8))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map over video groups
        group_results = list(tqdm(
            executor.map(_process_entry, entries),
            total=len(entries),
            desc="Processing videos"
        ))
    
    with open(os.path.join(dataset._dataset_path, "puls_new.json"), "w") as f:
        json.dump(group_results, f, indent=4)

if __name__ == "__main__":
    main()