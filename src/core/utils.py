import re
import sys
import os

if sys.platform == "win32":
    import pyttsx3

    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty("voices")
    tts_engine.setProperty("voice", voices[1].id)

    def generate_speech(text: str):
        tts_engine.save_to_file(text, "tmp/spoken.wav")
        tts_engine.runAndWait()

elif sys.platform == "linux":

    def generate_speech(text: str):
        command = f'espeak-ng -w tmp/spoken.wav "{text}"'
        os.system(command)


def split_text(text: str, max_tokens: int = 250):
    # Split text into sentences using a regex
    sentences = re.split(r"(?<=[.!?])+", text.strip())
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence
        sentence_length = len(sentence.split())

        # Check if adding this sentence exceeds the max_tokens limit
        if current_length + sentence_length > max_tokens:
            # Add the current chunk to the list if current chunk is not empty
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with the current sentence
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add the current sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add the final chunk if it's non-empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
