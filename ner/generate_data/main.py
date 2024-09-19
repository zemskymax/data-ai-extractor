import json
import re
import ollama

MODEL = "gemma2:27b"
# MODEL = "mistral:instruct"

MODEL_OPTIONS = {
    # defined in https://github.com/ollama/ollama/blob/main/docs/modelfile.md
    'num_ctx': 4096,        # default is 2048
    'num_predict': 1000,    # default is 128
    'temperature': 0.2,     # default is 0.8
    'top_k': 50,            # default is 40
    'top_p': 0.2,           # default is 0.9
    'repeat_penalty': 2.0,  # default is 1.1
    'seed': 17,             # default is 0
    'stop': ['<|end_of_turn|>']
}

def create_json_prompt_for_synthetic_data(**kwargs):
    # Building the initial part of the prompt
    prompt = """
        **Objective:**
        Produce realistic text passages that include clearly identified named entities. Each entity should be meticulously labeled according to its type for straightforward extraction.

        **Format Requirements:**
        - The output should be formatted in JSON, containing the text and the corresponding entities list.
        - Do not provide any comments.
        - Each entity in the text should be accurately marked and annotated in the 'entities' list.
        - Meticulously follow all the listed attributes.

        **Entity Annotation Details:**
        - All entity types must be in lowercase. For example, use "type" not "TYPE".
        - Entity types can be multiwords separate by space. For instance, use "entity type" rather than "entity_type".
        - Entities spans can be nested within other entities.
        - A single entity may be associated with multiple types. list them in the key "types".

        **Input:**
        <attribute_1="value1" attribute_2="value2" ...>
        
        **Output Schema:**
        {
        "text": "{text content}",
        "entities": [
            {"entity": "entity name", "types": ["type 1", "type 2", ...]},
            ...
        ]
        }

        **Here are some real world examples:**"""

    # Use dictionary comprehension to filter out 'n/a' values and to keep the code flexible
    attributes = {key: value for key, value in kwargs.items() if value != "n/a"}

    # Create a string of attributes for the <start> tag, excluding any 'n/a' values
    attributes_string = " ".join([f'{key}="{value}"' for key, value in attributes.items()])

    # Adding the dynamically created attributes string to the prompt
    prompt += f"""
        <{attributes_string}> """

    return prompt

def generate(**kwargs):
    prompt = create_json_prompt_for_synthetic_data(**kwargs)
    print(prompt)

    res = ollama.generate(MODEL, prompt=prompt, stream=False, options=MODEL_OPTIONS, keep_alive="1h")

    output_text = str(res["response"]).strip()
    print(f"output text: {output_text}")
    # return json.loads(outputs[0].outputs[0].text)


if __name__ == "__main__":
    generate(language="english", types_of_text="detailed job ads", sector="machine learning", country="france")

