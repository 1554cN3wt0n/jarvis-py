import requests
from bs4 import BeautifulSoup
import numpy as np


def get_wikipedia_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all paragraphs
    paragraphs = soup.find_all("p")

    # Extract text from paragraphs
    text = []
    for paragraph in paragraphs:
        text.append(paragraph.get_text())

    # Join the text content
    # return '\n'.join(text)

    # Return all paragraphs
    return text


def split_paragraphs(raw_text: str):
    # Split the raw text into paragraphs
    paragraphs = raw_text.split("\n")

    # Remove any leading/trailing whitespace from each paragraph
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    return paragraphs


def read_and_split_paragraphs(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content into paragraphs
    paragraphs = split_paragraphs(content)

    return paragraphs
