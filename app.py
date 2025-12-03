import cv2
import numpy as np
import streamlit as st

from PIL import Image
from openai import OpenAI
from oct_model import load_model_and_scaler, predict_oct, CLASSES

# page config
st.set_page_config(page_title="OCT Scan Assistant", layout="wide")

# OpenAI API integration
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# load the model and the scaler from oct_model.py
@st.cache_resource
def get_model_and_scaler():
    return load_model_and_scaler()

model, scaler = get_model_and_scaler()

# Custom Sidebar using CSS
st.sidebar.markdown(
"""
    <style>
        /* Use Radio Button as the default concept */
        .stRadio > div {
            gap: 0px;
        }

        .stRadio > div > label {
            width: 100%;
            cursor: pointer;
            padding: 10px 0px 10px 5px;
            font-size: 20px;
            border-radius: 4px;
            background-color: transparent;
        }
                        
        .stRadio > div > label:hover {
            padding-left: 12px;
            background-color: #3a3a3a;
        }

        .stRadio > div > label > div:first-child {
            display: none;
        }
        
        .stRadio > div > label:has(input:checked) {
            font-weight: bold;
            padding-left: 12px;
            background-color: #5b5b5b;
        }
        
        .stRadio > label {
            display: none;
        }

    </style>
"""
,unsafe_allow_html=True)

# Keep track of page
if "page" not in st.session_state:
    st.session_state.page = "Home"

#list of items at the sidebar
st.sidebar.title("Menu")
menu_items = ["Home", "About", "Model Summary", "OCT Scan's Prediction + AI Assistant"]

# Use radio buttons for navigation with their own unique keys
selected_page = st.sidebar.radio("Menu", menu_items, key="nav_radio", label_visibility="collapsed")

# Check so that if the same button is click, it wont trigger again
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# -- Home --
if st.session_state.page == "Home":
    st.title("OCT Scan Assistant for Rural Clinic")
    st.markdown(
        """
        ### Welcome, Doctor!

        This tool is designed to help rural or resource-limited clinics by providing:

        - **OCT B-Scan Image Classification using SVM**
        - **Simple AI Assistant (LLM)** to explain the prediction
        - **Model Performance Summary**

        ### How to Use:
        1. Go To **OCT Scan's Prediction + AI Assistant
        2. Upload an OCT B-Scan image
        3. It will then, give:
            - Comparison of Original vs Preprocessed OCT
            - Predicted Disease Class
            - LLM Explanation

        > **DISCLAIMER:**
        This tool is strictly used for academic purposes, and should not be a standardized for clinical diagnosis.
        """
    )

# -- About --
elif st.session_state.page == "About":
    st.title("About this Tools")
    st.markdown(
        """
        ### The Goal
        To Support rural clinic's by offering them a tool and an AI assistant that works by doing :
        
        - Image Preprocessing
        - Feature Extraction
        - SVM-based OCT Classification
        - Simple LLM Explanation and Contextualization

        ### Technology Used
        - Python + Streamlit
        - SVM Machine Learning Model
        - OpenAI GPT-4o-mini for assistant responses

        ### Important Notice
        This tool does **not** provide a true medical diagnosis, only in giving a second opinion/decision on the retinal disease class.
        Please do consult a specialist ophtalmologist for actual treatment procedure.
        """
    )

# -- Model Summary --
elif st.session_state.page == "Model Summary":
    st.title("Model Summary & Evaluation")

    # Simple Statistic Summary
    MODEL_METRICS = {
        "Overall Accuracy": "75%",
        "Validation Accuracy": "74.6 %",
        "Test Accuracy": "73.8%",
        "Dataset Size" : "18400 images"
    }

    st.subheader("Key Metrics")
    for k, v in MODEL_METRICS.items():
        st.write(f"**{k}:** {v}")

    st.markdown("---")
    st.subheader("Classification Report")

    st.text("Classification TBD.")

# OCT Prediction and LLM
elif st.session_state.page == "OCT Scan's Prediction + AI Assistant":
    st.title("OCT Scan's Prediction and AI Assistant")

    uploaded_file = st.file_uploader("Upload an OCT B-Scan image", type=["png","jpg","jpeg"])

    pred_label = None
    img_resized = None
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("L") #-> Grayscale
        img_np = np.array(pil_image)

        # resize the image, following the architecture
        img_resized = cv2.resize(img_np, (224, 224))

        # predict
        pred_label, enhanced = predict_oct(img_resized, model, scaler)

        # Show image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original OCT")
            st.image(img_resized, clamp=True, channels="GRAY", use_container_width=True)
        
        with col2:
            st.subheader("Preprocessed OCT")
            st.image(enhanced, clamp=True, channels="GRAY", use_container_width=True)

        # prediction summary
        st.markdown("---")
        st.markdown("### Model Prediction")

        st.markdown(
            f"""
            **Most Likely Class :**
            <span style="font-size: 26px; font-weight: 700;">{pred_label}</span>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "This is an estimation using the SVM model, not actual medical diagnosis.\n"
            "Please interpret together with clinical findings."
        )

    else:
        st.info("Please upload an OCT image!")

    # -- LLM Part --
    st.markdown("---")
    st.markdown("### AI Assistant - Second Opinion (Explanation and Definition)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    if pred_label is None:
        st.chat_input("Upload an image first!", disabled=True)
    else:
        user_msg = st.chat_input("Ask about this OCT result...")
        if user_msg:
            # add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.chat_message("user"):
                st.markdown(user_msg)
        
            system_prompt = (
                "You are an AI Assistant helping doctors in rural areas to interpret an OCT B-Scan."
                "Classification results. You are NOT a doctor nor a diagnostic system."
                "Your job:\n"
                "- Explain the model's predicted disease class in simple terms.\n"
                "- Mention possible characteristic for the predicted disease class.\n"
                "- Be Cautious and emphasize uncertainty.\n"
                "- Always include a disclaimer that this is not a diagnosis and the doctor must make their own judgement for the treatment.\n"
                "- Do NOT give treatment prescription.\n"
            )

            context_str = f"The model's predicted retinal disease class is: {pred_label}.\n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_str + "\nDoctor's Question: " + user_msg},
            ]
            
            # Call the LLM
            with st.chat_message("assistant"):
                with st.spinner("Thinking.."):
                    completion = client.chat.completions.create(
                        model = "gpt-4o-mini",
                        messages=messages,
                    )
                    answer = completion.choices[0].message.content
                    st.markdown(answer)

            # Save and display the assistant messages
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
st.sidebar.markdown("---")
st.sidebar.caption("Created for academic purposes.")






