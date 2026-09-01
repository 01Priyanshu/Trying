import streamlit as st
import google.generativeai as genai
import chromadb
from PIL import Image
import pypdf
import io
import base64
from streamlit_javascript import st_javascript

# --- 1. SETUP & CONFIG ---
st.set_page_config(layout="wide", page_title="CSE 205 IA Assistant (Live Monitor)")

if "GEMINI_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_KEY"]
else:
    st.error("Please add GEMINI_KEY to your Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# --- 2. CUSTOM KNOWLEDGE BASE (RAG) ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")["embedding"] for text in input]

client = chromadb.PersistentClient(path="./cse205_kb")
embedding_fn = GeminiEmbeddingFunction()
collection = client.get_or_create_collection(name="solutions", embedding_function=embedding_fn)

# --- 3. THE "LIVE LINK" JAVASCRIPT ---
def get_live_screen_frame():
    # This JS opens the browser's screen picker and returns the current frame as a DataURL
    js_code = """
    (async () => {
        if (!window.screenStream) {
            window.screenStream = await navigator.mediaDevices.getDisplayMedia({ video: { cursor: "always" }, audio: false });
        }
        const video = document.createElement('video');
        video.srcObject = window.screenStream;
        await video.play();
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/png');
    })()
    """
    return st_javascript(js_code)

# --- 4. SIDEBAR: KNOWLEDGE MANAGEMENT ---
with st.sidebar:
    st.title("📂 Knowledge Base")
    st.info("Upload instructor PDFs once to enable retrieval.")
    module_name = st.selectbox("Assign to Module", [f"Module {i}" for i in range(1, 15)] + ["Labs"])
    uploaded_pdfs = st.file_uploader("Upload Official PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Index PDFs"):
        if uploaded_pdfs:
            with st.spinner("Indexing PDFs..."):
                for pdf in uploaded_pdfs:
                    reader = pypdf.PdfReader(pdf)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            collection.upsert(
                                documents=[text], 
                                metadatas=[{"source": pdf.name, "page": i+1, "module": module_name}], 
                                ids=[f"{pdf.name}_{i}"]
                            )
            st.success("Indexing Complete!")

# --- 5. MAIN IA INTERFACE ---
st.title("CSE 205 IA Assistant")
st.write("Link your screen to automatically analyze what you are seeing on Zoom.")

col_screen, col_analysis = st.columns([1.2, 1])

with col_screen:
    st.subheader("🖥️ Live Monitor")
    
    # Trigger the JS Screen Capture
    raw_image_data = get_live_screen_frame()
    
    if raw_image_data and len(raw_image_data) > 100:
        # Convert the JS string to an image Gemini can read
        base64_str = raw_image_data.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        current_frame = Image.open(io.BytesIO(img_bytes))
        
        st.image(current_frame, caption="Monitoring your screen...", use_column_width=True)
        analyze_btn = st.button("⚡ ANALYZE CURRENT VIEW", type="primary", use_container_width=True)
    else:
        st.warning("Click the 'Link' button below and select your Zoom window or Entire Screen.")
        st.button("🔗 LINK LIVE SCREEN", use_container_width=True)
        analyze_btn = False

# --- 6. AUTOMATED ANALYSIS PIPELINE ---
if analyze_btn and current_frame:
    with st.spinner("🤖 Identifying Activity & Retrieving Solution..."):
        
        # Phase 1: Identify Activity from Screen
        id_prompt = "Identify the Activity Number, Challenge Number, or Lab Name from this screen (e.g. 'Activity 4.10.1')."
        id_res = model.generate_content([id_prompt, current_frame])
        search_terms = id_res.text
        
        # Phase 2: Search PDFs
        results = collection.query(query_texts=[search_terms], n_results=1)
        
        if not results['documents'][0]:
            with col_analysis:
                st.error(f"Solution for '{search_terms}' not found in Knowledge Base. Please upload the relevant PDFs.")
        else:
            official_sol = results['documents'][0][0]
            meta = results['metadatas'][0][0]
            
            # Phase 3: Compare Screen to Solution
            final_prompt = f"""
            Compare the code on the screen to the Official Solution.
            OFFICIAL SOLUTION: {official_sol}
            
            TASK:
            1. Find the exact logic or syntax error on the screen.
            2. Reference {meta['source']} p.{meta['page']}.
            
            OUTPUT:
            [IA DIAGNOSIS]
            Direct explanation of the bug. No guessing.
            
            [SAY TO STUDENT]
            2-3 sentence human-like hint leading them to the fix.
            """
            
            analysis = model.generate_content([final_prompt, current_frame])
            
            with col_analysis:
                st.subheader("IA Guidance")
                st.success(f"Verified against {search_terms}")
                st.markdown(analysis.text)
                
                with st.expander("View Official Code"):
                    st.code(official_sol, language="java")
