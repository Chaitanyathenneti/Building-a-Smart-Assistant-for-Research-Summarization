import streamlit as st
from pdfminer.high_level import extract_text
from transformers import pipeline
from textwrap import wrap

# --------------------- Page Configuration ---------------------
st.set_page_config(page_title="Smart Research Assistant", layout="wide")

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.title("Smart Research Assistant")
    st.markdown("""
    Use this tool to:
    - Upload a PDF or TXT file  
    - Auto-generate a summary  
    - Ask custom questions  
    - Try the challenge quiz  
    """)
    st.markdown("---")
    st.caption("Built for EzWorks GenAI Assignment")

# --------------------- Caching Models ---------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_question_generator():
    return pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

summarizer = load_summarizer()
qa_pipeline = load_qa_pipeline()
question_generator = load_question_generator()

# --------------------- Upload Section ---------------------
st.title("Smart Research Assistant")
st.write("Upload a research document (PDF or TXT), get a summary, ask questions, and test your understanding.")

with st.expander("Upload Your Document", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

def extract_file_text(file):
    if file.type == "application/pdf":
        return extract_text(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

document_text = ""
if uploaded_file:
    document_text = extract_file_text(uploaded_file)
    st.success("Document uploaded and text extracted successfully!")

# --------------------- Summary Section ---------------------
def split_text(text, max_chars=1000):
    return wrap(text, max_chars)

if document_text:
    st.subheader("Auto-Generated Summary")
    st.info("Summarizing the main points of your uploaded document...")

    text_chunks = split_text(document_text, max_chars=1000)
    combined_summary = ""

    for i, chunk in enumerate(text_chunks[:5]):
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        combined_summary += f"{i+1}. {summary[0]['summary_text']}\n\n"

    st.write(combined_summary.strip())
    st.markdown("---")

# --------------------- Ask Anything Section ---------------------
if document_text:
    st.subheader("Ask Anything About the Document")
    st.markdown("Type a question to get an answer based strictly on the document.")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Ask your question")
    with col2:
        submit = st.button("Get Answer")

    if submit and question:
        with st.spinner("Searching for the answer..."):
            answer = qa_pipeline(question=question, context=document_text[:2000])
            st.success(f"Answer: {answer['answer']}")
            st.caption(f"Confidence Score: {round(answer['score'] * 100, 2)}%")
    st.markdown("---")

# --------------------- Challenge Me Mode ---------------------
if document_text:
    st.subheader("Challenge Me: Auto Quiz")
    st.markdown("Click below to generate 3 logic/comprehension questions based on your document.")

    if "questions" not in st.session_state:
        st.session_state.questions = []
        st.session_state.user_answers = []

    if st.button("Generate Questions"):
        prompt = f"generate questions: {document_text[:1000]}"
        output = question_generator(prompt, max_length=256, num_return_sequences=1)[0]['generated_text']
        questions_only = output.strip().split('\n')
        questions_only = [q.strip() for q in questions_only if q.strip()]
        if not questions_only:
            questions_only = ["Unable to generate clear questions."]
        st.session_state.questions = questions_only[:3]
        st.session_state.user_answers = [""] * len(st.session_state.questions)

    if st.session_state.questions:
        for i, q in enumerate(st.session_state.questions):
            st.text(f"Q{i+1}: {q}")
            st.session_state.user_answers[i] = st.text_input(f"Your Answer", key=f"ans_{i}")

        if all(ans.strip() for ans in st.session_state.user_answers):
            if st.button("Evaluate Answers"):
                st.subheader("Evaluation Results")
                for i, (q, user_ans) in enumerate(zip(st.session_state.questions, st.session_state.user_answers)):
                    result = qa_pipeline(question=q, context=document_text[:2000])
                    correct = result["answer"]
                    score = round(result["score"] * 100, 2)

                    st.markdown(f"*Q{i+1}: {q}*")
                    st.write(f"Your Answer: {user_ans}")
                    st.write(f"Correct Answer: {correct}")
                    if user_ans.strip().lower() == correct.strip().lower():
                        st.success("Correct")
                    else:
                        st.error("Incorrect")
                    st.caption(f"Confidence Score: {score}%")
                    st.markdown("---")

# --------------------- Footer ---------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart Research Assistant – EzWorks GenAI Assignment")
