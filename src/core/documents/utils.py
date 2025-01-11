def split_paragraphs(raw_text: str):
    # Split the raw text into paragraphs
    paragraphs = raw_text.split("\n")

    # Remove any leading/trailing whitespace from each paragraph
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    return paragraphs
