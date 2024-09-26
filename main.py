import argparse
import fitz  # PyMuPDF
import os
import ollama
import re
from enum import Enum
from gliner import GLiNER


# global parameters
PROMPT = """
    **IDENTITY and PURPOSE**
    You are a specialist in data engineering with advanced expertise in human names, including a degree in onomastics. Your task is to extract all human names from the following text.

    **OUTPUT INSTRUCTIONS**
    - Ensure to extract and print each human name on a separate line.
    - Include all names; do not stop after the first.
    - Return **only** the human names. Do not include any additional text, explanations, or formatting tags (e.g., `</span></text><br/>`).
    - If no human names are found, return absolutely nothing — no text, no spaces, no comments.
    - Ensure the output is clean and contains only human names.
    - Do not provide any explanations.

    **EXAMPLES**
    John Smith
    Emma Williams
    Liam Brown

    **INPUT**
    TEXT: {input_text}
"""
MODEL = "gemma2"
MODEL_OPTIONS = {
    # defined in https://github.com/ollama/ollama/blob/main/docs/modelfile.md
    'num_ctx': 2048,        # default is 2048
    'num_predict': 512,     # default is 128
    'temperature': 0.2,     # default is 0.8
    'top_k': 5,             # default is 40
    'top_p': 0.1,           # default is 0.9
    'repeat_penalty': 2.0,  # default is 1.1
    'seed': 17,             # default is 0
    'stop': ['<|end_of_turn|>']
}
INPUT_FOLDER = "input"
INPUT_FILE_EXTENSION = ".pdf"

class ParsingMethod(Enum):
    none = 'none'
    llm = 'llm'
    ner = 'ner'

    def __str__(self):
        return self.value

class ReadingMethod(Enum):
    none = 'none'
    paragraph = 'paragraph'
    sentence = 'sentence'

    def __str__(self):
        return self.value

class NerType(Enum):
    none = 'none'
    base = 'base'
    tuned = 'tuned'

    def __str__(self):
        return self.value

##----------------------------------------##

def parse_text_llm(text):
    for input_text in text:
        print(f"input text: {input_text}")
        prompt = PROMPT.format(input_text=input_text)
        # print(f"complete prompt: {prompt}")

        res = ollama.generate(MODEL, prompt=prompt, stream=False, options=MODEL_OPTIONS, keep_alive="1h")
        # print("Response: " + str(res))
        output_text = str(res["response"]).strip()
        print(f"output text: {output_text}")
        print("--")

##----------------------------------------##

def parse_text_ner(text, ner_type):
    labels = ["first_name"]
    if ner_type == NerType.tuned:
        trained_model = GLiNER.from_pretrained("models/checkpoint-510", load_tokenizer=True, local_files_only=True)
    elif ner_type == NerType.base:
        trained_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    else:
        print("ERROR. no NER model type was chosen!")
        return

    for input_text in text:
        print(f"input text:\n{input_text}")
        entities = trained_model.predict_entities(input_text, labels, threshold=0.5)

        output_text = ""
        for ent in entities:
            print(ent["text"], "=>", ent["label"], "=>", ent["score"])
            if bool(output_text):
                output_text += "\n"
            output_text += ent["text"]

        print(f"output text:\n{output_text}")
        print("--")

##----------------------------------------##

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pm", "--parsing_method", required=True, type=ParsingMethod, choices=list(ParsingMethod), help='TODO')
    parser.add_argument('-rm', '--reading_method', required=True, type=ReadingMethod, choices=list(ReadingMethod), help='TODO')
    parser.add_argument('-nt', '--ner_type', required=True, type=NerType, choices=list(NerType), help='TODO')

    return parser.parse_args()

##----------------------------------------##

def main():
    args = parse_args()

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(INPUT_FILE_EXTENSION):
            print(f"Process the {file_name} file.")
            full_input_path = os.path.join(INPUT_FOLDER, file_name)

            doc = fitz.open(full_input_path)

            print("*Start*")
            total_pages_to_read = 2
            for page_index, page in enumerate(doc.pages()):
                if page_index == 0:
                    print("-Skipping first page-")
                    continue
                elif page_index == 1:
                    print("-Skipping second page-")
                    continue
                elif page_index < (total_pages_to_read + 2):
                    print(f"-Reading the {page_index} page-")
                    input_text = []
                    if args.reading_method == ReadingMethod.paragraph:
                        paragraphs = page.get_text("blocks")

                        total_paragraphs_to_read = 3
                        for paragraph in paragraphs:
                            paragraph = paragraph[4].replace('\n', ' ').replace('\r', '').replace('  ', ' ')

                            words = paragraph.split()
                            words_counter = len(words)
                            # clear paragraphs under 2 words
                            if words_counter <= 2:
                                continue

                            # print(f"Paragraph has {words_counter} words.")
                            paragraph = " ".join(words)
                            # print(paragraph)

                            total_paragraphs_to_read -= 1
                            if total_paragraphs_to_read < 0:
                                break

                            input_text.append(paragraph)
                    elif args.reading_method == ReadingMethod.sentence:
                        sentences = re.findall(r'([^.!?]*[.!?])', page.get_text())

                        total_sentences_to_read = 10
                        for sentence in sentences:
                            # remove line breaks (new lines)
                            sentence = sentence.replace('\n', ' ').replace('\r', '').replace('  ', ' ').strip().lower()

                            words = sentence.split()
                            words_counter = len(words)
                            # clear sentences under 2 words
                            if words_counter <= 2:
                                continue

                            # print(f"Sentence has {words_counter} words.")
                            sentence = " ".join(words)
                            # print(sentence)

                            total_sentences_to_read -= 1
                            if total_sentences_to_read < 0:
                                break

                            input_text.append(sentence)
                    else:
                        print("ERROR. no reading method has been chosen!")
                        return

                    if args.parsing_method == ParsingMethod.llm:
                        parse_text_llm(input_text)
                    elif args.parsing_method == ParsingMethod.ner:
                        parse_text_ner(input_text, args.ner_type)
                    else:
                        print("ERROR. no parsing method has been chosen!")
                        return
                else:
                    print("*Stop*")
                    break


if __name__ == "__main__":
    main()
