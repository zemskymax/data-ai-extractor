import json
import os
import re
import random
import time
from vllm import LLM, SamplingParams
from names import *


# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# LLM_MODEL = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
LLM_MODEL = "neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit"
NUM_GPUs = 1


def save_data_to_file(data, full_file_path):
    """Saves the processed data to a file in a JSON format."""
    with open(full_file_path, 'w') as f:
        json.dump(data, f)

##----------------------------------------##

def create_prompt_for_synthetic_data_generation(**kwargs):
    """Creates a prompt that the input parameters. This prompt will be used to generate the synthetic data."""
    # Building the initial part of the prompt
    prompt = """
        **Objective:**
        Generate realistic text passages that include named entities. Each entity should be clearly identified and labeled with its type(s) for easy extraction.

        **Format Requirements:**
        - Output should be formatted in JSON and include both the text and a list of entities.
        - No additional comments or explanations should be included in the output.
        - Each entity must be accurately labeled and appear in the entities list.
        - Follow all attribute requirements exactly as specified.
        - Output only the JSON object.

        **Entity Annotation Details:**
        - All entity types must be in lowercase. For example, use "type" instead of "TYPE".
        - Entity types can contain multiple words, separated by spaces (e.g., "entity type", not "entity_type").
        - Nested entities are allowed (an entity can be within another entity).
        - An entity can be associated with multiple types. In such cases, list them under the "types" key.

        **Output Schema Example:**
        <start attribute_1="value1" attribute_2="value2" ...>
        {
        "text": "text content",
        "entities": [
            {"entity": "entity name", "types": ["type 1", "type 2", ...]},
            ...
        ]
        }
        <end>

        **Finish the following Schema:**"""

    # Use dictionary comprehension to filter out 'n/a' values and to keep the code flexible
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}

    # Create a string of attributes for the <start> tag, excluding any 'n/a' values
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])

    # Adding the dynamically created attributes string to the prompt
    prompt += f"""
        <start {attributes_string}>"""

    return prompt

##----------------------------------------##

def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

##----------------------------------------##

def extract_entities(data):
    all_examples = []

    for dt in data:
        # Attempt to extract entities; skip current record on failure
        try:
            tokens = tokenize_text(dt['text'])
            ents = [(k["entity"], k["types"]) for k in dt['entities']]
        except Exception as ex:
            print(f"Exception: {ex}!")
            continue

        spans = []
        for entity in ents:
            entity_tokens = tokenize_text(str(entity[0]))

            # Find the start and end indices of each entity in the tokenized text
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                    for el in entity[1]:
                        spans.append((i, i + len(entity_tokens) - 1, el.lower().replace('_', ' ')))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})

    return all_examples

##----------------------------------------##

def generate_from_prompts(prompt, llm, sampling_params):
    outputs = llm.generate(prompt, sampling_params)
    # print(f"> total outputs:\n {outputs}")

    all_outs = []
    for output in outputs:
        try:
            # print(f"-> raw output:\n {output.outputs[0].text.strip()}")
            js = json.loads(output.outputs[0].text.strip())
        except Exception as ex:
            print(f"Exception: {ex}!")
            continue

        # print(f"--> output as JSON:\n {js})")
        all_outs.append(js)

    return all_outs, extract_entities(all_outs)


OUTPUT_FOLDER = "data"
OUTPUT_FILE_SUFFIX = ".json"
NUM_SAMPLES = 5
if __name__ == "__main__":
    llm = LLM(model=LLM_MODEL, gpu_memory_utilization=0.8, max_model_len=1500, tensor_parallel_size=NUM_GPUs, seed=17, quantization="GPTQ")

    sampling_params = SamplingParams(top_k=100, max_tokens=1000, top_p=0.8, stop="<end>")

    all_outputs = []
    for i in range(NUM_SAMPLES):
        name = random.choice(FIRST_NAMES)
        surname = random.choice(SECOND_NAMES)

        prompt = create_prompt_for_synthetic_data_generation(language="english",
                                                       types_of_text="casual sentence",
                                                       name=name,
                                                       surname=surname)

        output, processed_output = generate_from_prompts(prompt, llm, sampling_params)
        all_outputs.append(processed_output[0])

        print(f"Final response: {output}")
        print(f"Processed output {processed_output[0]}")

    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    file_name = time.strftime("%Y%m%d-%H%M%S")
    save_data_to_file(all_outputs, dir_name + "/" + OUTPUT_FOLDER + "/" + file_name + OUTPUT_FILE_SUFFIX)

    print("Done")
