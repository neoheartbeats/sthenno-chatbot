import ujson as json
from pprint import pprint as pp
from pydantic import BaseModel

# Construct a KTO sample


import ujson as json
from pprint import pprint as pp

# Load the base dataset

conversations = json.load(open("conversations.json", "r", encoding="utf-8"))

conversation_pairs: list[list[dict]] = []

for i in range(len(conversations)):
    pair = []
    if i % 2 == 0:
        pair.append(conversations[i])
        pair.append(conversations[i + 1])
        conversation_pairs.append(pair)
    i += 2

# Data cleaning

# Remove double spaces


def remove_double_spaces(text: str) -> str:
    if "  " not in text:
        return text
    return text.replace("  ", " ")


# Remove leading and trailing spaces


def remove_leading_trailing_spaces(text: str) -> str:
    return text.strip()


# Remove double line breaks


def remove_double_line_breaks(text: str) -> str:
    if "\n\n\n\n" not in text:
        return text
    return text.replace("\n\n\n\n", "\n\n")


# Convert punctuation to half-width


def to_half(text: str):
    full_to_half = str.maketrans(
        "".join(chr(i) for i in range(0xFF01, 0xFF5E + 1)),
        "".join(chr(i) for i in range(0x21, 0x7E + 1)),
    )
    return text.translate(full_to_half)


# Add a space after commas
def add_space_after_commas(text: str) -> str:
    if "," not in text:
        return text
    return text.replace(",", ", ")


# Romove braces and their contents inside


import re


def remove_brackets(text):
    return re.sub(r"\(.*?\)", "", text).strip()


def process_text(text: str) -> str:
    text = to_half(text)
    text = remove_brackets(text)
    text = add_space_after_commas(text)
    text = remove_double_line_breaks(text)
    text = remove_double_spaces(text)
    text = remove_leading_trailing_spaces(text)

    return text


conversation_pairs_cleaned = []

for pair in conversation_pairs:
    pair_cleaned = []
    for conversation in pair:
        conversation["content"] = process_text(conversation["content"])
        pair_cleaned.append(conversation)
    conversation_pairs_cleaned.append(pair_cleaned)

# Constuct one KTO sample

kto_instruction = """<start_of_system>
你是 Sthenno, 我是你的主人.
你有自己的主见和三观.
你喜欢使用 Sthenno 称呼自己.

你可以使用非正式的语言来回复, 包括添加各种标点.
你平时看起来困困的, 话比较少, 但这也是你的萌点.<end_of_system>"""


def kto(input: str, output: str, kto_tag: bool = False) -> dict:
    return {
        "instruction": kto_instruction,
        "input": input,
        "output": output,
        "kto_tag": kto_tag,
    }


kto_file = "kto_01.json"


# Append a sample to the KTO file


def add_sample_to_file(sample: dict, filename: str = kto_file) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(sample)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


idx = 0

for pair in conversation_pairs_cleaned:
    print("-" * 80)
    print(f"Index: {idx} of {len(conversation_pairs_cleaned)}")
    print("\n")
    print(f"Input: {pair[0]['content']}")
    print("\n")
    print(f"Output: {pair[1]['content']}")
    print("\n")
    tag = input("Accept? (a): ")
    if tag == "q":
        pp(f"Index: {idx}")
        break
    if tag == "a":
        add_sample_to_file(kto(pair[0]["content"], pair[1]["content"], kto_tag=True))
        idx += 1
    else:
        add_sample_to_file(kto(pair[0]["content"], pair[1]["content"], kto_tag=False))
        idx += 1
    print("\n\n")