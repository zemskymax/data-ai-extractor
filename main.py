import os
import re
from tabula import read_pdf
import fitz  # PyMuPDF
import ollama

# URL = "/home/maxpc/Downloads/War and Peace (Leo Tolstoy).pdf"
# tabular_data = read_pdf(URL, pages="all")
# print(tabular_data[0].head(5))

PROMPT = """
    IDENTITY and PURPOSE   
    You are a specialist in data engineering with advanced expertise in human names, including a degree in onomastics. Your task is to extract all human names from the following text.

    OUTPUT INSTRUCTIONS
    - Extract and list each human name on a separate line.
    - Include all names; do not stop after the first.
    - Return **only** the human names. Do not include any additional text, explanations, or formatting tags (e.g., `</span></text><br/>`).
    - If no human names are found, return absolutely nothing—no text, no spaces, no comments.
    - Ensure the output is clean and contains only human names.

    INPUT
    TEXT: {input_text}
"""

read_by_chapter = True

def parse_text(input_text):
    model = "gemma2"
    # model = "phi3.5"
    # model = "llama3"

    model_options = {
        # defined in https://github.com/ollama/ollama/blob/main/docs/modelfile.md
        'num_ctx': 2048,        # default is 2048
        'num_predict': 256,     # default is 128
        'temperature': 0.0,     # default is 0.8
        'top_k': 10,            # default is 40
        'top_p': 0.3,           # default is 0.9
        'repeat_penalty': 2.0,  # default is 1.1
        'seed': 17,             # default is 0
        'stop': ['<|eot_id|>']
    }

    for text in input_text:
        print("--> " + text)
        prompt = PROMPT.replace("{input_text}", text)
        print(prompt)

        res = ollama.generate(model, prompt=prompt, stream=True, options=model_options, keep_alive="1h")
        # print("Response: " + str(res))
        output_text = str(res["response"]).strip()
        print(f"output text: {output_text}")
        print("")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            print(file_name)
            full_input_path = os.path.join(input_folder, file_name)

            # Open the PDF document using PyMuPDF
            doc = fitz.open(full_input_path)

            # input_text = ""
            for page_index, page in enumerate(doc.pages()):
                print(f"Page: {page_index}")
                if page_index == 0:
                    print("-0-")
                    continue
                elif page_index == 1:
                    print("-1-")
                    continue
                elif page_index > 4:
                    break
                
                if page_index == 2:
                    input_text = []
                    text = page.get_text()
                    # read text by chapter
                    if read_by_chapter:
                        # read text by page & remove extra spaces
                        text = text.replace("  ", "")
                        # print(text)
                        words = text.split()
                        # words_counter = len(words)
                        chapter = " ".join(words)
                        input_text.append(chapter)
                    # read text by sentence
                    else:
                        # TODO. sentences without the last character (? / . / !). has to be added
                        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

                        count = 11
                        for sentence in sentences:
                            # remove line breaks
                            sentence = sentence.replace('\n', ' ').replace('\r', '').replace('  ', ' ').strip()
                            count -= 1
                            if count < 1:
                                break
                            
                            input_text.append(sentence)

                    # print(f"Input text: {input_text}. Text length: {words_counter}")

                    parse_text(input_text)