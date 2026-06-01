from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os, uuid, shutil, json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import requests

app = FastAPI()

# --- Config ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_PREFIX = "user_"
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant client
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Dummy user database
users = {"demo@synapse.ai": "password123"}
sessions = {}

# --- Helper functions ---
def create_user_collection(user_id):
    collection_name = f"{COLLECTION_PREFIX}{user_id}"
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    return collection_name

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def index_document(user_id, file_content, filename):
    os.makedirs("uploads", exist_ok=True)
    tmp_path = f"uploads/{uuid.uuid4()}_{filename}"
    with open(tmp_path, "wb") as f:
        f.write(file_content)

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_path)
    elif filename.endswith(".txt"):
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Unsupported file type")

    os.remove(tmp_path)

    chunks = chunk_text(text)
    points = []
    for i, chunk in enumerate(chunks):
        embedding = EMBEDDER.encode(chunk).tolist()
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk, "source": filename}
        ))

    collection = create_user_collection(user_id)
    qdrant.upsert(collection_name=collection, points=points)
    return {"chunks_indexed": len(chunks)}

def ask_question(user_id, question):
    collection_name = f"{COLLECTION_PREFIX}{user_id}"
    if not qdrant.collection_exists(collection_name):
        return "You haven't uploaded any documents yet."

    q_embedding = EMBEDDER.encode(question).tolist()
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=q_embedding,
        limit=3
    )
    context = "\n\n".join([hit.payload["text"] for hit in hits])

    prompt = f"""Use the following context to answer the question at the end. If you don't know, just say you don't know.
Context:
{context}

Question: {question}
Answer:"""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if resp.status_code == 200:
        return resp.json().get("response", "Error generating answer.")
    else:
        return "Could not reach LLM."

# --- The HTML page (embedded directly) ---
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>Synapse</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
        #chat-box { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .message { margin: 5px 0; }
        .user { color: blue; }
        .assistant { color: green; }
        #login-form, #main-area { display: none; }
    </style>
</head>
<body>
    <h1>Synapse</h1>

    <div id="login-form">
        <h2>Login</h2>
        <input type="email" id="email" placeholder="Email">
        <input type="password" id="password" placeholder="Password">
        <button onclick="login()">Login</button>
        <p id="login-error" style="color:red"></p>
    </div>

    <div id="main-area">
        <button onclick="logout()">Logout</button>
        <h2>Upload Document</h2>
        <input type="file" id="file-input">
        <button onclick="upload()">Upload</button>
        <p id="upload-status"></p>

        <h2>Chat with your documents</h2>
        <div id="chat-box"></div>
        <input type="text" id="question" placeholder="Ask a question" style="width:80%">
        <button onclick="ask()">Send</button>
    </div>

    <script>
        async function login() {
            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;
            const form = new FormData();
            form.append("email", email);
            form.append("password", password);
            const resp = await fetch("/login", {method: "POST", body: form});
            if (resp.ok) {
                document.getElementById("login-form").style.display = "none";
                document.getElementById("main-area").style.display = "block";
            } else {
                document.getElementById("login-error").textContent = "Invalid credentials";
            }
        }

        async function logout() {
            await fetch("/logout");
            document.getElementById("login-form").style.display = "block";
            document.getElementById("main-area").style.display = "none";
        }

        async function upload() {
            const fileInput = document.getElementById("file-input");
            const file = fileInput.files[0];
            if (!file) return alert("Select a file");
            const form = new FormData();
            form.append("file", file);
            const resp = await fetch("/upload", {method: "POST", body: form});
            const data = await resp.json();
            document.getElementById("upload-status").textContent = `Indexed ${data.chunks_indexed} chunks.`;
        }

        async function ask() {
            const question = document.getElementById("question").value;
            if (!question) return;
            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += `<div class="message user"><b>You:</b> ${question}</div>`;
            document.getElementById("question").value = "";

            const form = new FormData();
            form.append("question", question);
            const resp = await fetch("/ask", {method: "POST", body: form});
            const data = await resp.json();
            chatBox.innerHTML += `<div class="message assistant"><b>Synapse:</b> ${data.answer}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Show login by default
        document.getElementById("login-form").style.display = "block";
    </script>
</body>
</html>"""

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    if email in users and users[email] == password:
        session_token = str(uuid.uuid4())
        sessions[session_token] = email
        response = JSONResponse({"status": "ok"})
        response.set_cookie(key="session", value=session_token)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    session_token = request.cookies.get("session")
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=403, detail="Not logged in")
    user_id = sessions[session_token]
    content = file.file.read()
    result = index_document(user_id, content, file.filename)
    return result

@app.post("/ask")
def ask(request: Request, question: str = Form(...)):
    session_token = request.cookies.get("session")
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=403, detail="Not logged in")
    user_id = sessions[session_token]
    answer = ask_question(user_id, question)
    return {"answer": answer}

@app.get("/logout")
def logout(request: Request):
    session_token = request.cookies.get("session")
    if session_token:
        sessions.pop(session_token, None)
    return HTMLResponse("<script>window.location.href='/';</script>")