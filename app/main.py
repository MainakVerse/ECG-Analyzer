import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import scipy.io
from src.visualization import plot_ecg

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='🫀 ECG Classification',
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide',
    initial_sidebar_state="expanded"
)

# Custom CSS for beautification
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E63946;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #457B9D;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #F1FAEE;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #A8DADC;
        color: #1D3557;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #F1FAEE;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-highlight {
        color: #E63946;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .footer-text {
        text-align: center;
        color: #1D3557;
        margin-top: 2rem;
    }
    .stSidebar {
        background-color: #F1FAEE;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Create tabs for different sections of the app
tabs = st.tabs(["📊 ECG Classification", "💬 Ask the Cardio"])

#---------------------------------#
# Data preprocessing and Model building

@st.cache_data
def read_ecg_preprocessing(uploaded_ecg):
    FS = 300
    maxlen = 30*FS

    uploaded_ecg.seek(0)
    mat = scipy.io.loadmat(uploaded_ecg)
    mat = mat["val"][0]

    uploaded_ecg = np.array([mat])

    X = np.zeros((1,maxlen))
    uploaded_ecg = np.nan_to_num(uploaded_ecg) # removing NaNs and Infs
    uploaded_ecg = uploaded_ecg[0,0:maxlen]
    uploaded_ecg = uploaded_ecg - np.mean(uploaded_ecg)
    uploaded_ecg = uploaded_ecg/np.std(uploaded_ecg)
    X[0,:len(uploaded_ecg)] = uploaded_ecg.T # padding sequence
    uploaded_ecg = X
    uploaded_ecg = np.expand_dims(uploaded_ecg, axis=2)
    return uploaded_ecg

model_path = 'models/weights-best.hdf5'
classes = ['Normal','Atrial Fibrillation','Other','Noise']

@st.cache_resource
def get_model(model_path):
    model = load_model(f'{model_path}')
    return model

@st.cache_resource
def get_prediction(data, _model):
    prob = _model(data)
    ann = np.argmax(prob)
    return classes[ann], prob

# Visualization --------------------------------------
@st.cache_resource
def visualize_ecg(ecg, FS):
    fig = plot_ecg(uploaded_ecg=ecg, FS=FS)
    return fig

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar:
    st.image("https://api.iconify.design/openmoji/anatomical-heart.svg?width=100", width=100)
    st.markdown("<h2 style='text-align: center; color: #1D3557;'>ECG Analysis Tool</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### 1. Upload your ECG")
    uploaded_file = st.file_uploader("Upload your ECG in .mat format", type=["mat"])

    st.markdown("<hr>", unsafe_allow_html=True)

    file_gts = {
        "A00001": "Normal",
        "A00002": "Normal",
        "A00003": "Normal",
        "A00004": "Atrial Fibrilation",
        "A00005": "Other",
        "A00006": "Normal",
        "A00007": "Normal",
        "A00008": "Other",
        "A00009": "Atrial Fibrilation",
        "A00010": "Normal",
        "A00015": "Atrial Fibrilation",
        "A00205": "Noise",
        "A00022": "Noise",
        "A00034": "Noise",
    }
    
    valfiles = [
        'None',
        'A00001.mat','A00010.mat','A00002.mat','A00003.mat',
        "A00022.mat", "A00034.mat",'A00009.mat',"A00015.mat",
        'A00008.mat','A00006.mat','A00007.mat','A00004.mat',
        "A00205.mat",'A00005.mat'
    ]

    if uploaded_file is None:
        st.markdown("### 2. Or use a file from the validation set")
        pre_trained_ecg = st.selectbox(
            'Select a sample ECG',
            valfiles,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".mat","")))})' if ".mat" in x else x,
            index=1,
        )
        if pre_trained_ecg != "None":
            f = open("data/validation/"+pre_trained_ecg, 'rb')
            if not uploaded_file:
                uploaded_file = f
    else:
        st.info("Remove the file above to demo using the validation set.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='footer-text'>Made by <a href='https://github.com/MainakVerse'>Mainak</a></div>", unsafe_allow_html=True)

#---------------------------------#
# Main panel - Tab 1: ECG Classification
with tabs[0]:
    st.markdown("<h1 class='main-header'>🫀 ECG Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Detect Atrial Fibrillation, Normal Rhythm, Other Rhythm, or Noise from your ECG</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if uploaded_file is not None:
        # Initialize model
        model = get_model(f'{model_path}')
        
        col1, col2 = st.columns([0.55, 0.45])

        with col1:  # visualize ECG
            st.markdown("### Visualize ECG")
            with st.spinner("Processing ECG data..."):
                ecg = read_ecg_preprocessing(uploaded_file)
                fig = visualize_ecg(ecg, FS=300)
                st.pyplot(fig, use_container_width=True)

        with col2:  # classify ECG
            st.markdown("### Model Predictions")
            with st.spinner(text="Running Model..."):
                pred, conf = get_prediction(ecg, model)
            
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"<h3>ECG classified as <span class='result-highlight'>{pred}</span></h3>", unsafe_allow_html=True)
            
            pred_confidence = conf[0, np.argmax(conf)]*100
            st.markdown(f"<p>Confidence: <b>{pred_confidence:.1f}%</b></p>", unsafe_allow_html=True)
            
            st.markdown("#### Probability Distribution")
            
            # Create a bar chart for the confidence levels
            conf_data = {classes[i]: float(conf[0,i]*100) for i in range(len(classes))}
            chart_data = {"Rhythm Type": list(conf_data.keys()), "Confidence (%)": list(conf_data.values())}
            
            st.bar_chart(chart_data, x="Rhythm Type", y="Confidence (%)", use_container_width=True)
            
            # Create a table with detailed confidence levels
            st.markdown("#### Detailed Results")
            mkd_pred_table = [
                "| Rhythm Type | Confidence |",
                "| --- | --- |"
            ]
            for i in range(len(classes)):
                mkd_pred_table.append(f"| {classes[i]} | {conf[0,i]*100:3.1f}% |")
            mkd_pred_table = "\n".join(mkd_pred_table)
            st.markdown(mkd_pred_table)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Include interpretation info
            if pred == "Atrial Fibrillation":
                st.info("📌 Atrial Fibrillation is characterized by irregular and rapid heart rhythm. This condition increases the risk of stroke and heart failure.")
            elif pred == "Normal":
                st.success("✅ Your ECG shows a normal heart rhythm pattern. Regular check-ups are still recommended for heart health monitoring.")
            elif pred == "Other":
                st.warning("⚠️ The ECG shows an abnormal rhythm that is not classified as Atrial Fibrillation. Further clinical assessment is recommended.")
            elif pred == "Noise":
                st.error("❗ The ECG contains too much noise for reliable interpretation. Consider retaking the ECG in a more controlled environment.")
    else:
        st.info("👈 Please upload an ECG file or select a sample from the sidebar to start.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://api.iconify.design/openmoji/anatomical-heart.svg?width=300", use_column_width=True)
            
#---------------------------------#
# Tab 2: Ask the Cardio
with tabs[1]:
    st.markdown("<h1 class='main-header'>💬 Ask the Cardio</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your AI assistant for ECG and heart health questions</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your cardiology assistant. I can answer questions about ECGs, heart rhythms, and cardiovascular health. How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to generate responses about ECG and heart health
    def generate_cardio_response(prompt):
        # Dictionary of common ECG and cardiology questions and answers
        cardio_knowledge = {
            "atrial fibrillation": "Atrial fibrillation (AFib) is an irregular and often rapid heart rhythm that can increase risk of stroke, heart failure, and other heart-related complications. On an ECG, it's characterized by irregular R-R intervals and absence of P waves.",
            "normal ecg": "A normal ECG typically shows regular rhythm with P waves, QRS complexes, and T waves in sequence. The P-R interval is usually 0.12-0.20 seconds, QRS duration 0.06-0.10 seconds, and Q-T interval 0.36-0.44 seconds.",
            "heart rate": "Normal resting heart rate for adults ranges from 60-100 beats per minute (BPM). Athletes may have lower resting heart rates, sometimes as low as 40 BPM, which is usually not a concern.",
            "ecg leads": "A standard 12-lead ECG uses electrodes placed on the limbs and chest to record electrical activity from different angles. These include leads I, II, III, aVR, aVL, aVF (limb leads) and V1-V6 (chest leads).",
            "qt interval": "The QT interval represents ventricular depolarization and repolarization. A prolonged QT interval can indicate a risk for potentially dangerous arrhythmias like torsades de pointes.",
            "st elevation": "ST elevation on an ECG often indicates myocardial injury or infarction (heart attack). It represents damage to heart muscle and requires immediate medical attention.",
            "ecg interpretation": "ECG interpretation involves analyzing the regularity of rhythm, heart rate, P waves, PR interval, QRS complex, T waves, QT interval, and looking for any abnormal patterns or changes.",
            "heart block": "Heart blocks occur when electrical signals between the atria and ventricles are delayed or blocked. They can be first-degree (PR prolongation), second-degree (intermittent blocking), or third-degree (complete block).",
            "premature beats": "Premature beats can be atrial (PACs) or ventricular (PVCs). They appear as early beats on the ECG and are usually benign but can sometimes indicate underlying heart disease.",
            "ventricular tachycardia": "Ventricular tachycardia is a rapid heart rhythm starting in the ventricles. On ECG, it appears as wide QRS complexes at a rate typically >100 BPM. It can be life-threatening and requires immediate treatment."
        }
        
        # Check if any keywords match in the prompt
        response = "I don't have specific information about that in my cardiology knowledge base. Please ask something related to ECGs or heart conditions."
        
        # Simple keyword matching - could be enhanced with more sophisticated NLP
        for keyword, info in cardio_knowledge.items():
            if keyword.lower() in prompt.lower():
                response = info
                break
        
        # General queries about ECG
        if "what is" in prompt.lower() and "ecg" in prompt.lower():
            response = "An electrocardiogram (ECG or EKG) is a test that records the electrical activity of your heart. It shows how fast your heart beats and whether its rhythm is steady or irregular. ECGs are used to detect heart problems like arrhythmias, heart attacks, and structural abnormalities."
        
        # Queries about rhythm disorders
        if "rhythm disorder" in prompt.lower() or "arrhythmia" in prompt.lower():
            response = "Common cardiac rhythm disorders visible on ECG include:\n\n- Atrial fibrillation - irregular rhythm without P waves\n- Atrial flutter - rapid regular atrial activity with 'sawtooth' pattern\n- Ventricular tachycardia - rapid wide-complex rhythm\n- Bradycardia - abnormally slow heart rate (<60 BPM)\n- Heart blocks - disrupted conduction between chambers"
        
        return response
    
    # Chat input
    if prompt := st.chat_input("Ask a question about ECGs or heart health"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_cardio_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    # Display some example questions
    with st.expander("Example questions you can ask"):
        st.markdown("""
        - What does a normal ECG look like?
        - What is atrial fibrillation and how does it appear on an ECG?
        - What causes an elevated ST segment?
        - What is the QT interval and why is it important?
        - How can I interpret different ECG leads?
        - What are the signs of ventricular tachycardia on an ECG?
        - How does a heart block appear on an ECG?
        - What's the difference between PVCs and PACs?
        - How are arrhythmias classified?
        - What heart conditions can be diagnosed with an ECG?
        """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-top: 20px;'>
        <p style='color: #721c24; margin: 0;'><strong>Important Disclaimer:</strong> This AI assistant provides general information only and is not a substitute for professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='footer-text'>Made for Machine Learning in Healthcare with Streamlit</div>", unsafe_allow_html=True)
