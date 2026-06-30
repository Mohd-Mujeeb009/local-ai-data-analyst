# 🧠 Local AI Data Analyst

A **ChatGPT-style AI chatbot** that helps you analyze data, PDFs, and images conversationally using **Groq API** (LLaMA 3.3 70B + LLaMA 3.2 Vision).  
**Free API key** — no credit card required.

---

## ✨ Features

- 💬 ChatGPT-style conversational interface
- 📊 Analyze CSV & Excel files with natural language
- 📈 Automatic charts and tables based on AI responses
- 📄 PDF document analysis and Q&A
- 🖼️ Image understanding via LLaMA 3.2 Vision (charts, screenshots, photos)
- 🧠 Conversation memory (seamless follow-ups)
- ⚡ Blazing fast inference via Groq
- 🔑 Free API key — no credit card needed

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **AI Provider** | Groq (free tier) |
| **Text Model** | LLaMA 3.3 70B Versatile |
| **Vision Model** | LLaMA 3.2 90B Vision |
| **Language** | Python 3.9+ |
| **Data** | Pandas, Matplotlib |
| **PDF** | pypdf |

---

## 📁 Project Structure

```
local-ai-data-analyst/
├── backend/
│   ├── __init__.py
│   ├── llama_client.py    # Groq API client (text + vision)
│   ├── prompt.py          # System prompt configuration
│   ├── context.py         # Data/PDF context builders
│   └── file_handler.py    # File loading & image encoding
├── frontend/
│   ├── __init__.py
│   ├── app.py             # Main Streamlit application
│   ├── state.py           # Session state management
│   ├── utils.py           # Visualization detection helpers
│   └── visualizer.py      # Chart, table & image renderers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed
- **Groq API key** (free) — get one at [console.groq.com](https://console.groq.com)

### 1. Get Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/local-ai-data-analyst.git
cd local-ai-data-analyst
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`.

### 6. Enter Your API Key

Paste your Groq API key in the sidebar when the app opens. That's it!

---

## 📖 Usage

1. **Enter your API key** in the sidebar
2. **Upload a file** (CSV, Excel, PDF, or image)
3. **Ask questions** in natural language via the chat input
4. **Get AI-powered insights** — the chatbot remembers your data and prior messages
5. **Auto-visualizations** — charts and tables appear automatically when relevant

### Example Questions

- *"What are the top 5 products by revenue?"*
- *"Show me the trend over the last 6 months"*
- *"Summarize this PDF document"*
- *"What does this chart show?"* (with an uploaded image)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
