# Text Extraction using Large Language Models

## Overview
This project leverages a Large Language Model (LLM) named Gemma to extract specific textual content, such as character names, from books provided as PDF files. The LLM is run locally using the Ollama framework, while the PyMuPDF library is used to parse and handle the PDF files.

## Features
- Local LLM Deployment: The LLM (Gemma) is hosted and executed locally, ensuring privacy and control over the data.
- PDF Parsing: Utilizes the PyMuPDF library for efficient and accurate PDF parsing.
- Text Extraction: Capable of extracting specific content, such as character names, from extensive texts.

## Dependencies
- Ollama: Library to run the Gemma LLM locally.
- PyMuPDF: For parsing and handling PDF files.

## Usage
- Prepare the PDF File: Ensure that your book is available in PDF format (in the input folder) 
- Run the main.py: Execute the script to parse the PDF and extract the desired content.
    -- Run the fine tuned Gliner model
    python3 main.py -pm ner -rm sentence -nt tuned
    -- Run the base Gliner model
    python3 main.py -pm ner -rm sentence -nt base
    -- Run the LLM 
    python3 main.py -pm llm -rm sentence -nt none