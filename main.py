import gradio as gr
from core.jarvis import JARVIS
from argparse import ArgumentParser, BooleanOptionalAction
from dotenv import load_dotenv

load_dotenv()

parser = ArgumentParser()

parser.add_argument(
    "--share", type=bool, action=BooleanOptionalAction, help="share gradio link"
)

args = parser.parse_args()


model = JARVIS()


def load_document(file, url):
    if url is not None and url != "":
        model.load_context_from_wiki(url)
        return "Document loaded. You can now ask questions."
    if file is not None:
        model.load_context_from_txt(file.name)
        return "Document loaded. You can now ask questions."
    return "No context was loaded"


# Function to generate answers based on the document
def ask_question(question):
    answer = model.answer(question)
    return question, answer


# Create the Gradio interface
file_input = gr.File(label="Upload Document")
ulr_input = gr.Textbox(
    lines=1, placeholder="Paste the Wikipedia URL here...", label="Wikipedia URL"
)
text_input = gr.Textbox(
    lines=2, placeholder="Enter your question here...", label="Your Question"
)
text_question_ref = gr.Textbox(label="Your question was:")
text_output = gr.Textbox(label="Possible Answers:")

iface = gr.Interface(
    fn=ask_question,
    inputs=[text_input],
    outputs=[text_question_ref, text_output],
    examples=[["What is physics?"], ["What is mathematics?"], ["What is chemistry?"]],
    title="Document Q&A Chatbot",
    description="Upload a document and ask questions about it.",
)

# Add the document upload interface
iface_upload = gr.Interface(
    fn=load_document,
    inputs=[file_input, ulr_input],
    outputs="text",
    examples=[
        [None, "https://en.wikipedia.org/wiki/Physics"],
        [None, "https://en.wikipedia.org/wiki/Mathematics"],
        [None, "https://en.wikipedia.org/wiki/Chemistry"],
    ],
    title="Upload Document",
    description="Upload a document to ask questions about it.",
)

# Combine both interfaces
app = gr.TabbedInterface([iface_upload, iface], ["Upload Document", "Ask Questions"])

app.launch(server_name="0.0.0.0", server_port=8080, share=args.share)
