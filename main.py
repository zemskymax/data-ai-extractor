import os
import re
import fitz  # PyMuPDF
import ollama


# global parameters
PROMPT = """
    IDENTITY and PURPOSE   
    You are a specialist in data engineering with advanced expertise in human names, including a degree in onomastics. Your task is to extract all human names from the following text.

    OUTPUT INSTRUCTIONS
    - Ensure to extract and print each human name on a separate line.
    - Include all names; do not stop after the first.
    - Return **only** the human names. Do not include any additional text, explanations, or formatting tags (e.g., `</span></text><br/>`).
    - If no human names are found, return absolutely nothing — no text, no spaces, no comments.
    - Ensure the output is clean and contains only human names.
    - Do not provide any explanations.

    EXAMPLE
    ```
    Olivia Brown
    Liam Johnson
    Noah Williams
    Emma Smith
    ```

    INPUT
    TEXT: {input_text}
"""

MODEL = "gemma2"
# MODEL = "phi3.5"
# MODEL = "llama3.1"

MODEL_OPTIONS = {
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

def parse_text(text):
    for input_text in text:
        print(f"input text: {input_text}")
        prompt = PROMPT.replace("{input_text}", input_text)
        # print(prompt)

        res = ollama.generate(MODEL, prompt=prompt, stream=False, options=MODEL_OPTIONS, keep_alive="1h")
        # print("Response: " + str(res))
        output_text = str(res["response"]).strip()
        print(f"output text: {output_text}")
        print("--")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    read_by_paragraph = True

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            print(f"Process the {file_name} file.")
            full_input_path = os.path.join(input_folder, file_name)

            doc = fitz.open(full_input_path)

            print("*Start*")
            total_pages_to_read = 3
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
                    # read text by chapter
                    if read_by_paragraph:
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
                    # read text by sentence
                    else:
                        # TODO. sentences without the last character (? / . / !). has to be added
                        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', page.get_text())

                        total_sentences_to_read = 10
                        for sentence in sentences:
                            # remove line breaks (new lines)
                            sentence = sentence.replace('\n', ' ').replace('\r', '').replace('  ', ' ').strip()

                            words = sentence.split()
                            words_counter = len(words)
                            # clear sentences under 2 words
                            if words_counter <= 2:
                                continue

                            print(f"Sentence has {words_counter} words.")
                            sentence = " ".join(words)
                            print(sentence)

                            total_sentences_to_read -= 1
                            if total_sentences_to_read < 0:
                                break

                            input_text.append(sentence)

                    parse_text(input_text)
                else:
                    print("*Stop*")
                    break
