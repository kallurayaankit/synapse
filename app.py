from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
import os, uuid, json, sqlite3, hashlib, math, csv, tempfile, time, base64, io, logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import requests
from tavily import TavilyClient
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest

# Optional pytesseract
try:
    import pytesseract
except ImportError:
    pytesseract = None

# --- Logging setup ---
logging.basicConfig(
    filename="synapse.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("synapse")

app = FastAPI()

# --- Prometheus metrics ---
CHAT_REQUESTS = Counter("synapse_chat_requests_total", "Total chat requests")
TOOL_CALLS = Counter("synapse_tool_calls_total", "Total tool invocations")
ERRORS = Counter("synapse_errors_total", "Total errors")
LATENCY = Histogram("synapse_chat_latency_seconds", "Chat latency in seconds")

# --- Database connection ---
DB = sqlite3.connect("synapse.db", check_same_thread=False)
DB.row_factory = sqlite3.Row

# --- Create tables ---
DB.execute("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password_hash TEXT
)""")
DB.execute("""CREATE TABLE IF NOT EXISTS agent_config (
    user_id INTEGER PRIMARY KEY,
    system_prompt TEXT DEFAULT 'You are a helpful assistant.',
    tools TEXT DEFAULT '',
    hf_token TEXT DEFAULT '',
    selected_model TEXT DEFAULT 'mistral'
)""")
# Add missing columns if they don't exist (safe to run repeatedly)
for col in [("hf_token", "TEXT DEFAULT ''"), ("selected_model", "TEXT DEFAULT 'mistral'")]:
    try:
        DB.execute(f"ALTER TABLE agent_config ADD COLUMN {col[0]} {col[1]}")
    except:
        pass

DB.execute("""CREATE TABLE IF NOT EXISTS fine_tuned_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    model_name TEXT,
    base_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)""")
DB.execute("""CREATE TABLE IF NOT EXISTS connectors (
    user_id INTEGER,
    service TEXT,
    connected INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, service)
)""")
DB.commit()

# --- Config ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_PREFIX = "user_"
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
sessions = {}

# --- Tavily client (optional) ---
TAVILY_CLIENT = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        TAVILY_CLIENT = TavilyClient(api_key=tavily_key)
except:
    pass

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

def extract_text_from_pdf(file_bytes):
    """Read PDF from raw bytes."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        text = f"[PDF extraction error: {e}]"
    return text

def index_document(user_id, file_content, filename):
    os.makedirs("uploads", exist_ok=True)
    if isinstance(file_content, bytes):
        file_bytes = file_content
    else:
        file_bytes = file_content.encode("utf-8")

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(file_bytes))
        if pytesseract:
            text = pytesseract.image_to_string(image)
        else:
            text = "[Image OCR not available – install pytesseract]"
    elif filename.endswith(".txt"):
        text = file_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    chunks = chunk_text(text)
    if not chunks:
        chunks = [text]
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
    logger.info(f"User {user_id} indexed {len(chunks)} chunks from {filename}")
    return {"chunks_indexed": len(chunks)}

def retrieve_context(user_id, question):
    collection_name = f"{COLLECTION_PREFIX}{user_id}"
    if not qdrant.collection_exists(collection_name):
        return ""
    q_embedding = EMBEDDER.encode(question).tolist()
    # Use query_points (the current Qdrant API)
    result = qdrant.query_points(
        collection_name=collection_name,
        query=q_embedding,
        limit=3
    )
    # result.points is a list of ScoredPoint objects
    hits = result.points
    return "\n\n".join([hit.payload["text"] for hit in hits])

# --- Database helpers ---
def get_user_id(email: str) -> int:
    row = DB.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    return row["id"] if row else None

def get_agent_config(user_id: int):
    row = DB.execute("SELECT * FROM agent_config WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return {"system_prompt": "You are a helpful assistant.", "tools": [], "hf_token": "", "selected_model": "mistral"}
    return {
        "system_prompt": row["system_prompt"],
        "tools": row["tools"].split(",") if row["tools"] else [],
        "hf_token": row["hf_token"] or "",
        "selected_model": row["selected_model"] or "mistral"
    }

def save_agent_config(user_id: int, system_prompt: str, tools_list: list, hf_token: str = "", selected_model: str = "mistral"):
    tools_str = ",".join(tools_list)
    DB.execute(
        "REPLACE INTO agent_config (user_id, system_prompt, tools, hf_token, selected_model) VALUES (?, ?, ?, ?, ?)",
        (user_id, system_prompt, tools_str, hf_token, selected_model)
    )
    DB.commit()
    logger.info(f"User {user_id} updated agent config")

def get_fine_tuned_models(user_id):
    rows = DB.execute("SELECT model_name, base_model FROM fine_tuned_models WHERE user_id = ?", (user_id,)).fetchall()
    return [{"model_name": row["model_name"], "base_model": row["base_model"]} for row in rows]

# --- Connector simulation ---
def mock_ingest_google_drive(user_id):
    sample_texts = [
        "Mock Google Drive: Q3 project roadmap – launch Synapse 2.0 by September.",
        "Google Drive: Meeting notes – discuss resource allocation and hiring.",
    ]
    for i, text in enumerate(sample_texts):
        index_document(user_id, text, f"gdrive_{i}.txt")

def mock_ingest_notion(user_id):
    sample_texts = [
        "Notion: Weekly standup – John is working on frontend, Sarah on backend.",
        "Notion: Product requirements for Synapse Phase 4 – add multi-modal and connectors.",
    ]
    for i, text in enumerate(sample_texts):
        index_document(user_id, text, f"notion_{i}.txt")

def mock_ingest_github(user_id):
    sample_texts = [
        "README.md: Synapse is an AI command center that can connect your data.",
        "Issue #42: Add multi-modal support for image and PDF uploads.",
    ]
    for i, text in enumerate(sample_texts):
        index_document(user_id, text, f"github_{i}.txt")

# --- Tool implementations ---
def web_search_tool(query: str) -> str:
    TOOL_CALLS.inc()
    if not TAVILY_CLIENT:
        return "Web search is not available (Tavily API key not set)."
    try:
        results = TAVILY_CLIENT.search(query, max_results=3)
        return "\n\n".join([f"{r['title']}: {r['content']}" for r in results.get("results", [])])
    except Exception as e:
        ERRORS.inc()
        logger.error(f"Web search error: {e}")
        return f"Search error: {e}"

def calculator_tool(expression: str) -> str:
    TOOL_CALLS.inc()
    allowed = {"math": math, "abs": abs, "round": round, "min": min, "max": max}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        ERRORS.inc()
        logger.error(f"Calculator error: {e}")
        return f"Calculator error: {e}"

def email_tool(to: str, subject: str, body: str) -> str:
    TOOL_CALLS.inc()
    with open("emails.log", "a") as f:
        f.write(f"TO: {to}\nSUBJECT: {subject}\nBODY: {body}\n---\n")
    return f"Email sent to {to} (simulated)."

# Tool registry
TOOLS = {
    "web_search": {
        "func": web_search_tool,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            }
        }
    },
    "calculator": {
        "func": calculator_tool,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression, e.g., '345 * 78'"}
                    },
                    "required": ["expression"]
                }
            }
        }
    },
    "email": {
        "func": email_tool,
        "schema": {
            "type": "function",
            "function": {
                "name": "email",
                "description": "Send a simulated email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email."},
                        "subject": {"type": "string", "description": "Email subject."},
                        "body": {"type": "string", "description": "Email body."}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        }
    }
}

# --- HTML pages ---
LOGIN_PAGE = """<!DOCTYPE html>
<html>
<head><title>Synapse</title>
<style>
    body { font-family: sans-serif; max-width: 400px; margin: 50px auto; }
    form { margin: 20px 0; }
</style>
</head>
<body>
    <h1>Synapse</h1>
    <h2>Login</h2>
    <form id="login-form">
        <input type="email" id="email" placeholder="Email" required><br><br>
        <input type="password" id="password" placeholder="Password" required><br><br>
        <button type="submit">Login</button>
    </form>
    <p id="login-error" style="color:red"></p>
    <hr>
    <h2>Register</h2>
    <form id="register-form">
        <input type="email" id="reg-email" placeholder="Email" required><br><br>
        <input type="password" id="reg-password" placeholder="Password" required><br><br>
        <button type="submit">Register</button>
    </form>
    <p id="reg-error" style="color:red"></p>
    <script>
        document.getElementById("login-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;
            const form = new FormData();
            form.append("email", email);
            form.append("password", password);
            const resp = await fetch("/login", {method: "POST", body: form});
            if (resp.ok) window.location.href = "/home";
            else document.getElementById("login-error").textContent = "Invalid credentials";
        });
        document.getElementById("register-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const email = document.getElementById("reg-email").value;
            const password = document.getElementById("reg-password").value;
            const form = new FormData();
            form.append("email", email);
            form.append("password", password);
            const resp = await fetch("/register", {method: "POST", body: form});
            if (resp.ok) {
                alert("Registration successful! Please log in.");
            } else {
                const data = await resp.json();
                document.getElementById("reg-error").textContent = data.detail || "Error";
            }
        });
    </script>
</body>
</html>"""

HOME_PAGE = """<!DOCTYPE html>
<html>
<head><title>Synapse</title>
<style>
    body { font-family: sans-serif; max-width: 900px; margin: auto; padding: 20px; }
    #chat-box { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
    .message { margin: 5px 0; }
    .user { color: blue; }
    .assistant { color: green; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 20px; }
</style>
</head>
<body>
    <nav>
        <a href="/home">Chat</a>
        <a href="/agent">Configure Agent</a>
        <a href="/finetune">Fine‑tune</a>
        <a href="/connectors">Connectors</a>
        <a href="/admin">Admin</a>
        <a href="/logout">Logout</a>
    </nav>
    <h1>Synapse</h1>
    <h2>Upload Document</h2>
    <input type="file" id="file-input">
    <button onclick="upload()">Upload</button>
    <p id="upload-status"></p>
    <h2>Chat</h2>
    <div id="chat-box"></div>
    <input type="text" id="question" placeholder="Ask a question" style="width:80%">
    <button onclick="send()">Send</button>
    <script>
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
        async function send() {
            const question = document.getElementById("question").value;
            if (!question) return;
            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += `<div class="message user"><b>You:</b> ${question}</div>`;
            document.getElementById("question").value = "";
            const form = new FormData();
            form.append("question", question);
            const resp = await fetch("/chat", {method: "POST", body: form});
            const data = await resp.json();
            chatBox.innerHTML += `<div class="message assistant"><b>Synapse:</b> ${data.answer}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>"""

AGENT_PAGE = """<!DOCTYPE html>
<html>
<head><title>Agent Configuration</title>
<style>
    body { font-family: sans-serif; max-width: 600px; margin: auto; padding: 20px; }
    label { font-weight: bold; }
</style>
</head>
<body>
    <nav>
        <a href="/home">Chat</a>
        <a href="/agent">Configure Agent</a>
        <a href="/finetune">Fine‑tune</a>
        <a href="/connectors">Connectors</a>
        <a href="/admin">Admin</a>
        <a href="/logout">Logout</a>
    </nav>
    <h1>Configure Your Agent</h1>
    <form id="config-form">
        <label>System Prompt:</label><br>
        <textarea id="system_prompt" rows="4" cols="50">{{system_prompt}}</textarea><br><br>
        <label>Tools:</label><br>
        <input type="checkbox" name="tool" value="web_search" {{web_search_checked}}> Web Search<br>
        <input type="checkbox" name="tool" value="calculator" {{calculator_checked}}> Calculator<br>
        <input type="checkbox" name="tool" value="email" {{email_checked}}> Email Simulator<br><br>
        <label>Model:</label>
        <select id="model_select">
            <option value="mistral">Base Model (Mistral)</option>
            {{custom_model_options}}
        </select><br><br>
        <button type="submit">Save</button>
    </form>
    <p id="status"></p>
    <script>
        const currentModel = "{{selected_model}}";
        const select = document.getElementById("model_select");
        for (let i = 0; i < select.options.length; i++) {
            if (select.options[i].value === currentModel) {
                select.selectedIndex = i;
                break;
            }
        }
        document.getElementById("config-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const systemPrompt = document.getElementById("system_prompt").value;
            const tools = [];
            document.querySelectorAll('input[name="tool"]:checked').forEach(cb => tools.push(cb.value));
            const selectedModel = document.getElementById("model_select").value;
            const resp = await fetch("/agent", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({system_prompt: systemPrompt, tools: tools, selected_model: selectedModel})
            });
            if (resp.ok) document.getElementById("status").textContent = "Configuration saved!";
            else document.getElementById("status").textContent = "Error saving.";
        });
    </script>
</body>
</html>"""

FINETUNE_PAGE = """<!DOCTYPE html>
<html>
<head><title>Fine‑tune Model</title>
<style>
    body { font-family: sans-serif; max-width: 700px; margin: auto; padding: 20px; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 20px; }
</style>
</head>
<body>
    <nav>
        <a href="/home">Chat</a>
        <a href="/agent">Configure Agent</a>
        <a href="/finetune">Fine‑tune</a>
        <a href="/connectors">Connectors</a>
        <a href="/admin">Admin</a>
        <a href="/logout">Logout</a>
    </nav>
    <h1>Fine‑tune Your Model</h1>
    <p>Select a base model. Your uploaded documents will be used for training.</p>
    <label>Base Model:</label>
    <select id="base_model">
        <option value="mistralai/Mistral-7B-Instruct-v0.1">Mistral 7B Instruct</option>
        <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
    </select><br><br>
    <label>HF Token (write):</label>
    <input type="password" id="hf_token" placeholder="hf_..."><br>
    <small>We'll save this in your agent config so you don't have to re-enter it.</small><br><br>
    <button onclick="startFinetune()">Start Fine‑Tuning</button>
    <div id="status"></div>
    <script>
        async function startFinetune() {
            const baseModel = document.getElementById("base_model").value;
            const hfToken = document.getElementById("hf_token").value;
            if (!hfToken) { alert("Please enter your HF token."); return; }
            document.getElementById("status").textContent = "Creating project...";
            const resp = await fetch("/finetune", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({base_model: baseModel, hf_token: hfToken})
            });
            const data = await resp.json();
            if (data.error) {
                document.getElementById("status").textContent = "Error: " + data.error;
                return;
            }
            document.getElementById("status").textContent = data.message;
        }
    </script>
</body>
</html>"""

CONNECTORS_PAGE = """<!DOCTYPE html>
<html>
<head><title>Connectors</title>
<style>
    body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 20px; }
    .connector { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
</style>
</head>
<body>
    <nav>
        <a href="/home">Chat</a>
        <a href="/agent">Configure Agent</a>
        <a href="/finetune">Fine‑tune</a>
        <a href="/connectors">Connectors</a>
        <a href="/admin">Admin</a>
        <a href="/logout">Logout</a>
    </nav>
    <h1>Connected Apps</h1>
    <div class="connector">
        <h3>Google Drive</h3>
        <button onclick="connect('google_drive')">Connect (simulate)</button>
    </div>
    <div class="connector">
        <h3>Notion</h3>
        <button onclick="connect('notion')">Connect (simulate)</button>
    </div>
    <div class="connector">
        <h3>GitHub</h3>
        <button onclick="connect('github')">Connect (simulate)</button>
    </div>
    <div id="status"></div>
    <script>
        async function connect(service) {
            const resp = await fetch(`/connect/${service}`, {method: "POST"});
            const data = await resp.json();
            document.getElementById("status").textContent = data.message;
        }
    </script>
</body>
</html>"""

ADMIN_PAGE = """<!DOCTYPE html>
<html>
<head><title>Admin Dashboard</title>
<style>
    body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
    .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
    pre { background: #000; color: #0f0; padding: 10px; overflow-x: auto; height: 400px; overflow-y: scroll; }
</style>
</head>
<body>
    <nav>
        <a href="/home">Chat</a>
        <a href="/agent">Configure Agent</a>
        <a href="/finetune">Fine‑tune</a>
        <a href="/connectors">Connectors</a>
        <a href="/admin">Admin</a>
        <a href="/logout">Logout</a>
    </nav>
    <h1>📊 Admin Dashboard</h1>
    <div class="metric"><strong>Total Chat Requests:</strong> <span id="chats">Loading...</span></div>
    <div class="metric"><strong>Total Tool Calls:</strong> <span id="tools">Loading...</span></div>
    <div class="metric"><strong>Total Errors:</strong> <span id="errors">Loading...</span></div>
    <div class="metric"><strong>Avg Latency (last 10):</strong> <span id="latency">Loading...</span></div>
    <h2>Recent Logs</h2>
    <pre id="logs">Loading...</pre>
    <script>
        async function refresh() {
            const resp = await fetch("/admin-data");
            const data = await resp.json();
            document.getElementById("chats").textContent = data.chat_requests;
            document.getElementById("tools").textContent = data.tool_calls;
            document.getElementById("errors").textContent = data.errors;
            document.getElementById("latency").textContent = data.avg_latency.toFixed(3) + " s";
            document.getElementById("logs").textContent = data.logs;
        }
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>"""

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return LOGIN_PAGE

@app.post("/register")
def register(email: str = Form(...), password: str = Form(...)):
    if get_user_id(email):
        raise HTTPException(status_code=400, detail="Email already registered")
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    DB.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
    DB.commit()
    logger.info(f"New user registered: {email}")
    return {"status": "ok", "message": "Registration successful. You can now log in."}

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = DB.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if not user or user["password_hash"] != hashlib.sha256(password.encode()).hexdigest():
        ERRORS.inc()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    session_token = str(uuid.uuid4())
    sessions[session_token] = user["id"]
    response = JSONResponse({"status": "ok"})
    response.set_cookie(key="session", value=session_token)
    logger.info(f"User {user['id']} logged in")
    return response

def get_current_user(request: Request):
    session_token = request.cookies.get("session")
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=403, detail="Not logged in")
    return sessions[session_token]

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    get_current_user(request)
    return HOME_PAGE

@app.get("/agent", response_class=HTMLResponse)
def agent_page(request: Request):
    user_id = get_current_user(request)
    config = get_agent_config(user_id)
    tools = config["tools"]
    custom_models = get_fine_tuned_models(user_id)
    options_html = ""
    for m in custom_models:
        options_html += f'<option value="{m["model_name"]}">{m["model_name"]} ({m["base_model"]})</option>\n'
    html = AGENT_PAGE
    html = html.replace("{{system_prompt}}", config["system_prompt"])
    html = html.replace("{{web_search_checked}}", "checked" if "web_search" in tools else "")
    html = html.replace("{{calculator_checked}}", "checked" if "calculator" in tools else "")
    html = html.replace("{{email_checked}}", "checked" if "email" in tools else "")
    html = html.replace("{{custom_model_options}}", options_html)
    html = html.replace("{{selected_model}}", config["selected_model"])
    return HTMLResponse(html)

@app.post("/agent")
async def save_agent(request: Request):
    user_id = get_current_user(request)
    data = await request.json()
    system_prompt = data.get("system_prompt", "")
    tools_list = data.get("tools", [])
    hf_token = data.get("hf_token", "")
    selected_model = data.get("selected_model", "mistral")
    save_agent_config(user_id, system_prompt, tools_list, hf_token, selected_model)
    return {"status": "ok"}

@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    user_id = get_current_user(request)
    content = file.file.read()
    filename = file.filename
    return index_document(user_id, content, filename)

@app.post("/chat")
async def chat(request: Request, question: str = Form(...)):
    user_id = get_current_user(request)
    context = retrieve_context(user_id, question)
    config = get_agent_config(user_id)
    system_prompt = config["system_prompt"]
    active_tool_names = config["tools"]
    model_to_use = "mistral"

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Relevant document context:\n{context}"})
    messages.append({"role": "user", "content": question})

    tool_schemas = [TOOLS[tool]["schema"] for tool in active_tool_names if tool in TOOLS]

    ollama_req = {
        "model": model_to_use,
        "messages": messages,
        "tools": tool_schemas,
        "stream": False
    }
    start = time.time()
    resp = requests.post(OLLAMA_CHAT_URL, json=ollama_req)
    latency = time.time() - start
    LATENCY.observe(latency)
    CHAT_REQUESTS.inc()

    if resp.status_code != 200:
        ERRORS.inc()
        logger.error("Ollama chat error")
        return {"answer": "Error communicating with LLM."}

    result = resp.json()

    while result.get("message", {}).get("tool_calls"):
        message = result["message"]
        tool_calls = message["tool_calls"]
        tool_results = []
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            if isinstance(raw_args, str):
                func_args = json.loads(raw_args)
            else:
                func_args = raw_args
            if func_name in TOOLS:
                tool_func = TOOLS[func_name]["func"]
                tool_output = tool_func(**func_args)
                tool_results.append((tc["id"], func_name, tool_output))

        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": tool_calls
        })
        for tc_id, fname, output in tool_results:
            messages.append({"role": "tool", "content": output, "tool_call_id": tc_id})

        ollama_req["messages"] = messages
        resp = requests.post(OLLAMA_CHAT_URL, json=ollama_req)
        if resp.status_code != 200:
            ERRORS.inc()
            return {"answer": "Error during tool processing."}
        result = resp.json()

    final_answer = result.get("message", {}).get("content", "No response.")
    logger.info(f"Chat response for user {user_id}: {final_answer[:50]}...")
    return {"answer": final_answer}

# --- Fine‑tuning routes ---
@app.get("/finetune", response_class=HTMLResponse)
def finetune_page(request: Request):
    get_current_user(request)
    return FINETUNE_PAGE

@app.post("/finetune")
async def start_finetune(request: Request):
    user_id = get_current_user(request)
    data = await request.json()
    base_model = data["base_model"]
    hf_token = data["hf_token"]
    config = get_agent_config(user_id)
    save_agent_config(user_id, config["system_prompt"], config["tools"], hf_token)
    fake_job_id = f"local-{uuid.uuid4().hex[:8]}"
    DB.execute("INSERT INTO fine_tuned_models (user_id, model_name, base_model) VALUES (?, ?, ?)",
               (user_id, f"synapse-{user_id}-{fake_job_id}", base_model))
    DB.commit()
    logger.info(f"Simulated fine-tune job for user {user_id}: {base_model}")
    return {"status": "started", "job_id": fake_job_id,
            "message": "Local fine‑tuning job simulated (no real GPU used)."}

# --- Connectors routes ---
@app.get("/connectors", response_class=HTMLResponse)
def connectors_page(request: Request):
    get_current_user(request)
    return CONNECTORS_PAGE

@app.post("/connect/{service}")
def connect_service(service: str, request: Request):
    user_id = get_current_user(request)
    DB.execute("REPLACE INTO connectors (user_id, service, connected) VALUES (?, ?, 1)", (user_id, service))
    DB.commit()
    if service == "google_drive":
        mock_ingest_google_drive(user_id)
    elif service == "notion":
        mock_ingest_notion(user_id)
    elif service == "github":
        mock_ingest_github(user_id)
    else:
        raise HTTPException(status_code=400, detail="Unknown service")
    logger.info(f"User {user_id} connected to {service}")
    return {"status": "ok", "message": f"Connected to {service} and ingested sample documents."}

# --- Admin & Metrics routes ---
@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    get_current_user(request)
    return ADMIN_PAGE

@app.get("/admin-data")
def admin_data(request: Request):
    get_current_user(request)
    # Read last 100 lines of the log file
    try:
        with open("synapse.log", "r") as f:
            lines = f.readlines()
            last_lines = lines[-100:]
            log_text = "".join(last_lines)
    except:
        log_text = "Log file not found."

    # Compute average latency from the histogram
    total_latency = 0.0
    count_latency = 0
    for metric in LATENCY.collect():
        for sample in metric.samples:
            if sample.name == "synapse_chat_latency_seconds_sum":
                total_latency = sample.value
            elif sample.name == "synapse_chat_latency_seconds_count":
                count_latency = int(sample.value)
    avg_latency = total_latency / count_latency if count_latency > 0 else 0.0

    return {
        "chat_requests": int(CHAT_REQUESTS._value.get()),
        "tool_calls": int(TOOL_CALLS._value.get()),
        "errors": int(ERRORS._value.get()),
        "avg_latency": avg_latency,
        "logs": log_text
    }

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/logout")
def logout(request: Request):
    session_token = request.cookies.get("session")
    if session_token:
        sessions.pop(session_token, None)
    return HTMLResponse("<script>window.location.href='/';</script>")