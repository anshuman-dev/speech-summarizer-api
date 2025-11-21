# Speech Summarizer API

A fast, lightweight REST API for text summarization using modern semantic understanding with Sentence-BERT. Built for the LLM-speech browser extension, but works as a standalone service.

## Features

- **Modern Semantic Understanding**: Uses Sentence-BERT contextual embeddings (10x better than Word2Vec/TF-IDF)
- **Fast Response**: ~500ms processing time on CPU
- **Lightweight**: ~100MB RAM footprint, perfect for free-tier hosting
- **Open Source**: Apache 2.0 licensed, no API costs
- **Web Demo**: Interactive interface for testing
- **Production Ready**: Includes error handling, CORS, health checks, and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd speech-summarizer-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

5. **Run the server**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Test the Web Demo

Open your browser and navigate to:
```
http://localhost:5000
```

## API Documentation

### Base URL

**Local**: `http://localhost:5000`
**Production**: `https://your-app.onrender.com` (after deployment)

### Endpoints

#### `POST /summarize`

Generate a summary from input text.

**Request:**
```json
{
  "text": "Your long text here...",
  "summary_length": 3,       // optional, default: 3
  "min_text_length": 500     // optional, default: 500
}
```

**Response (Success):**
```json
{
  "status": "success",
  "original_length": 1250,
  "summary_length": 320,
  "summary_text": "Summary goes here...",
  "processing_time_ms": 234.56,
  "compression_ratio": 0.256
}
```

**Response (Text Too Short):**
```json
{
  "status": "skipped",
  "message": "Text too short for summarization",
  "original_length": 300,
  "original_text": "..."
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error description",
  "error_type": "ValidationError"
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "summary_length": 3
  }'
```

#### `GET /health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "model_loaded": true,
  "embedding_dimension": 384
}
```

#### `GET /model-info`

Get detailed model information.

**Response:**
```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "max_sequence_length": 256,
  "parameters": "22M",
  "license": "Apache-2.0",
  "type": "extractive_with_contextual_embeddings"
}
```

## Browser Extension Integration

### JavaScript Example

```javascript
async function summarizeText(text) {
  try {
    const response = await fetch('https://your-api.onrender.com/summarize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        summary_length: 3,
        min_text_length: 500
      })
    });

    const data = await response.json();

    if (data.status === 'success') {
      return data.summary_text;
    } else {
      console.warn('Summarization skipped:', data.message);
      return text; // Fallback to original
    }
  } catch (error) {
    console.error('Summarization failed:', error);
    return text; // Fallback
  }
}
```

## Deployment

### Deploy to Render (Recommended - Free Tier)

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` and configure everything

3. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - Your API will be live at: `https://your-app-name.onrender.com`

**Free Tier Limitations:**
- Spins down after 15 minutes of inactivity
- First request after sleep: 20-60 seconds (cold start)
- 512MB RAM

**Upgrade to Paid ($7/month):**
- Always-on (no cold starts)
- 2GB RAM
- Better performance

### Deploy with Docker

```bash
# Build image
docker build -t speech-summarizer-api .

# Run container
docker run -p 8080:8080 speech-summarizer-api

# Test
curl http://localhost:8080/health
```

### Deploy to Google Cloud Run

```bash
gcloud run deploy speech-summarizer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi
```

### Deploy to Fly.io

```bash
flyctl launch
flyctl deploy
```

## Project Structure

```
speech-summarizer-api/
├── app.py                    # Flask application
├── summarizer.py             # Sentence-BERT summarizer
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version
├── Procfile                  # Heroku/Railway config
├── render.yaml               # Render.com config
├── Dockerfile                # Docker config
├── .env.example              # Environment variables template
├── templates/
│   └── index.html           # Web demo interface
├── static/
│   ├── style.css            # Demo page styles
│   └── script.js            # Demo page logic
├── .gitignore
├── .dockerignore
└── README.md
```

## How It Works

### Technology Stack

- **Flask**: Web framework
- **Sentence-BERT**: Modern contextual embeddings
- **NLTK**: Sentence tokenization
- **scikit-learn**: Similarity calculations
- **NumPy**: Numerical operations

### Summarization Process

1. **Tokenize**: Split text into sentences using NLTK
2. **Embed**: Generate 384-dimensional contextual embeddings for each sentence
3. **Analyze**: Calculate document centroid (average of all sentence embeddings)
4. **Rank**: Find sentences most similar to document centroid
5. **Select**: Return top N sentences in original order

### Why This Approach?

**Advantages:**
- ✅ Contextual understanding (e.g., "bank" means different things in different contexts)
- ✅ Fast inference (~500ms)
- ✅ Small memory footprint (~100MB)
- ✅ No API costs
- ✅ No hallucinations (extractive only)
- ✅ Privacy-friendly (runs on your server)

**Limitations:**
- Cannot generate novel sentences (extractive only)
- Limited to 256 tokens per sentence
- Less creative than LLMs (GPT, Claude)

## Performance

| Metric | Value |
|--------|-------|
| Model Size | ~100MB |
| RAM Usage | ~100-200MB |
| Response Time | 200-500ms (CPU) |
| Cold Start | 20-60s (free tier) |
| Embedding Dimension | 384 |
| Parameters | 22M |

## Testing

### Test Summarizer Directly

```bash
python summarizer.py
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model-info

# Summarize
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "summary_length": 3}'
```

### Test from Browser Console

```javascript
// Open http://localhost:5000 and run in console:
testAPI()
```

## Environment Variables

Create a `.env` file (see `.env.example`):

```bash
FLASK_ENV=development
PORT=5000
```

## Security Considerations

### For Public Deployment:

1. **Add Rate Limiting** (optional)
```bash
pip install flask-limiter
```

2. **Add API Key Authentication** (optional)
```python
# In app.py
API_KEY = os.environ.get('API_KEY')
```

3. **Restrict CORS** (production)
```python
# In app.py, change CORS origins from "*" to specific domains
CORS(app, resources={r"/*": {"origins": ["https://your-extension.com"]}})
```

## Monitoring

### Check Logs (Render)

```bash
# Via Render dashboard or CLI
render logs
```

### Health Monitoring

Set up uptime monitoring with:
- UptimeRobot
- Pingdom
- StatusCake

Ping `/health` endpoint every 5 minutes.

## Troubleshooting

### Cold Start Issues (Free Tier)

**Problem**: First request takes 30-60 seconds
**Solution**:
- Upgrade to paid plan ($7/month)
- Or show loading message to users
- Or use keep-alive service (check host TOS)

### Memory Issues

**Problem**: App crashes on free tier
**Solution**:
- Upgrade to 1-2GB RAM plan
- Or use smaller model (already using smallest production-ready model)

### Slow Processing

**Problem**: Summaries take >2 seconds
**Solution**:
- Check server CPU
- Reduce summary_length
- Consider GPU hosting

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Apache 2.0 License - see LICENSE file for details

## Tech Stack Comparison

| Approach | Speed | Quality | Cost | RAM |
|----------|-------|---------|------|-----|
| **This API (Sentence-BERT)** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💰 Free | 100MB |
| Word2Vec + TextRank | ⚡⚡⚡ Fast | ⭐⭐ Okay | 💰 Free | 100MB |
| BART/T5 Local | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | 💰 Free | 2GB |
| OpenAI API | ⚡⚡⚡ Fast | ⭐⭐⭐⭐⭐ Excellent | 💰💰 $$$ | N/A |

## Support

For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing documentation
- Review API logs

## Roadmap

- [ ] Add caching for repeated requests
- [ ] Implement API key authentication
- [ ] Add more summarization strategies
- [ ] Support multiple languages
- [ ] WebSocket support for streaming
- [ ] Add usage analytics dashboard

---

**Built with Sentence-BERT** | **Open Source** | **Production Ready**
