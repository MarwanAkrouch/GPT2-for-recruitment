import streamlit as st
import pandas as pd

from minGPT.mingpt.bpe import BPETokenizer

from predict import predict_single_pair
from  utils import load_model

model_name = 'gpt2_b1024_l12_h12_e768_nowd_lt0.4_ht0.75'

st.set_page_config(page_title="JD-CV Matcher", page_icon="🎯")

st.title("JD-CV Matching App 🎯")

# Initialize session state for resumes if not exists
if 'resumes' not in st.session_state:
    st.session_state.resumes = []

# Model loading (with caching)
@st.cache_resource
def get_model():
    return load_model(f"checkpoints/{model_name}", "best.pt")

model, device = get_model()

# to be able to use v0 models
try:
    sep_token = model.config.sep_token
    pad_token_id = model.config.pad_token_id
    thresholds = model.config.thresholds
    label_map=model.config.label_map
except AttributeError:
    sep_token = '###'
    pad_token_id = 0
    thresholds = [0.35, 0.85]
    label_map={
            'No Fit': 0.0,
            'Potential Fit': 0.6,
            'Good Fit': 1.0
        }

# Input fields
st.subheader("Job Description")
jd_text = st.text_area("Enter the job description:", height=200)

# Resume input with multiple resume support
st.subheader("Resumes")

# Text input for new resume
new_resume = st.text_area("Enter a resume:", height=200, key="new_resume_input")

# Add resume button
if st.button("Add Resume"):
    if new_resume and new_resume.strip():
        st.session_state.resumes.append(new_resume)
        st.success("Resume added successfully!")
    else:
        st.error("Please enter a resume text")

# Display currently added resumes
if st.session_state.resumes:
    st.write("### Current Resumes:")
    for i, resume in enumerate(st.session_state.resumes, 1):
        st.write(f"{i}. Resume (Preview): {resume[:100]}...")

sep_token = '###'
pad_token_id = 0  # 50256

def get_category(score):
    if score < 0.35:
        return "No Fit", "🔴"
    elif score < 0.85:
        return "Potential Fit", "🟡"
    else:
        return "Good Fit", "🟢"


# Analyze all resumes button
if st.button("Analyze All Resumes"):
    if not jd_text:
        st.error("Please enter a job description first")
    elif not st.session_state.resumes:
        st.error("Please add at least one resume")
    else:
        # Create results dataframe
        results = []
        
        with st.spinner("Analyzing resumes..."):
            for i, cv_text in enumerate(st.session_state.resumes, 1):
                # Get prediction
                score = predict_single_pair(
                    model=model,
                    jd_text=jd_text,
                    cv_text=cv_text,
                    tokenizer=BPETokenizer(),
                    block_size=model.config.block_size,
                    device=device
                )
                
                category, emoji = get_category(score)
                
                results.append({
                    'Resume Number': i,
                    'Match Score': score,
                    'Category': category,
                    'Emoji': emoji
                })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df_sorted = results_df.sort_values('Match Score', ascending=False)
        
        # Display results
        st.subheader("Ranking of Resumes")
        st.dataframe(results_df_sorted)
        
        # Detailed visualization
        st.bar_chart(results_df_sorted.set_index('Resume Number')['Match Score'])

# Clear all resumes button
if st.button("Clear All Resumes"):
    st.session_state.resumes = []
    st.success("All resumes have been cleared")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Powered by GPT</p>
</div>
""", unsafe_allow_html=True)