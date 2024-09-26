import json
import os
import re
import random
import time
from vllm import LLM, SamplingParams
from constants import *


# global parameters
OUTPUT_FOLDER = "data"
OUTPUT_FILE_EXTENSION = ".json"
NUM_SAMPLES = 10
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

def extract_entities(data, entities):
    all_examples = []

    for dt in data:
        # Attempt to extract entities; skip current record on failure
        try:
            tokens = tokenize_text(dt['text'])
            # FIX. LLM's outputs are not consistent, overriding the categories manually
            # entities = [(k["entity"], k["types"]) for k in dt['entities']]
            all_entities = entities
        except Exception as ex:
            print(f"Exception (while extracting entities): {ex}!")
            continue

        spans = []
        for entity in all_entities:
            # print(f"::: {entity} :::")
            categories = []

            for category in entity[1]:
                if category.lower() == "name":
                    categories.append("first_name")
                elif category.lower() == "surname":
                    categories.append("last_name")

            entity_tokens = tokenize_text(str(entity[0]))

            # Find the start and end indices of each entity in the tokenized text
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                    for category in categories:
                        spans.append((i, i + len(entity_tokens) - 1, category.lower()))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})

    return all_examples

##----------------------------------------##

def generate_from_prompt(prompt, llm, sampling_params, entities):
    outputs = llm.generate(prompt, sampling_params)
    # print(f"> total outputs:\n {outputs}")

    all_outs = []
    for output in outputs:
        try:
            # print(f"-> raw output:\n {output.outputs[0].text.strip()}")
            js = json.loads(output.outputs[0].text.strip())
        except Exception as ex:
            print(f"Exception (while generating from prompt): {ex}!")
            continue

        # print(f"--> output as JSON:\n {js})")
        all_outs.append(js)

    return all_outs, extract_entities(all_outs, entities)


if __name__ == "__main__":
    llm = LLM(model=LLM_MODEL, gpu_memory_utilization=0.8, max_model_len=1500, tensor_parallel_size=NUM_GPUs, seed=17, quantization="GPTQ")

    sampling_params = SamplingParams(top_k=100, max_tokens=1000, top_p=0.8, temperature=0.6, stop="<end>")

    all_outputs = []
    for text_type in TEXT_TYPES:
        for j in range(NUM_SAMPLES):
            random.seed(time.process_time())
            name = random.choice(FIRST_NAMES)
            surname = random.choice(SECOND_NAMES)

            prompt = create_prompt_for_synthetic_data_generation(language="english",
                                                                types_of_text = text_type,
                                                                name=name,
                                                                surname=surname)

            entities = [(name, ["name"]),(surname, ["surname"])]
            output, processed_output = generate_from_prompt(prompt, llm, sampling_params, entities)
            # all_outputs.append(processed_output)
            all_outputs += processed_output

            print(f"Final response: {output}")
            print(f"Processed output {processed_output}")

    print(all_outputs)
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    file_name = time.strftime("%Y%m%d-%H%M%S")
    save_data_to_file(all_outputs, dir_name + "/" + OUTPUT_FOLDER + "/" + file_name + OUTPUT_FILE_EXTENSION)

    print("Done")
