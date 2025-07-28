import fitz  # PyMuPDF
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
import torch

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file"""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def initialize_qa_model():
    """Load BERT model and tokenizer for question answering"""
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def get_answer(question, context, qa_pipeline):
    """Get answer from context for a given question"""
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    # Load PDF and extract context ===
    pdf_path = input("Enter path to your PDF file: ").strip()
    context = extract_text_from_pdf(pdf_path)
    print("\nPDF loaded successfully!\n")
    # the pdf is supposed to be in the same folder as the program file

    # Initialize QA system
    qa_system = initialize_qa_model()
    print("✅ BERT Question Answering model initialized.\n")

    print("Ask your questions or type 'exit' to stop.\n")

    while True:
        question = input("Q: ")
        if question.lower() in ("exit", "quit"):
            print("Exiting QA system.")
            break
        try:
            answer = get_answer(question, context, qa_system)
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"Error answering your question: {e}\n")

if __name__ == "__main__":
    main()
