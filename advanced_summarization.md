# Advanced Text Summarization: Bridging the Gap with LLMs

## Comprehensive Summarization Strategy

### 1. Advanced Hybrid Summarization Approach

```python
import numpy as np
import networkx as nx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import spacy

class AdvancedSummarizer:
    def __init__(self, language_model='en_core_web_sm'):
        # Load advanced NLP models
        self.nlp = spacy.load(language_model)
        nltk.download('punkt')
        
    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        - Sentence segmentation
        - Named entity recognition
        - Lemmatization
        """
        doc = self.nlp(text)
        
        # Enhanced sentence preprocessing
        sentences = [sent.text for sent in doc.sents]
        processed_sentences = []
        
        for sentence in sentences:
            # Lemmatization and removal of less meaningful sentences
            processed_sent = self.nlp(sentence)
            lemmatized_sent = ' '.join([token.lemma_ for token in processed_sent 
                                        if not token.is_stop and token.pos_ not in ['PUNCT', 'SYM']])
            
            # Filter out very short or uninformative sentences
            if len(lemmatized_sent.split()) > 3:
                processed_sentences.append(lemmatized_sent)
        
        return processed_sentences
    
    def semantic_similarity(self, sentences):
        """
        Advanced semantic similarity using Word2Vec
        """
        # Tokenize sentences
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Train Word2Vec model
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, 
                         window=5, min_count=1, workers=4)
        
        # Create sentence vectors
        sentence_vectors = []
        for sentence in tokenized_sentences:
            vector = np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
            sentence_vectors.append(vector)
        
        return sentence_vectors
    
    def advanced_textrank(self, sentences, sentence_vectors):
        """
        Enhanced TextRank with semantic similarity
        """
        # Create similarity graph
        similarity_graph = nx.Graph()
        
        for i in range(len(sentences)):
            similarity_graph.add_node(i)
        
        # Connect sentences based on semantic similarity
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                cosine_similarity = np.dot(sentence_vectors[i], sentence_vectors[j]) / (
                    np.linalg.norm(sentence_vectors[i]) * np.linalg.norm(sentence_vectors[j])
                )
                
                # Add edge if semantic similarity is above threshold
                if cosine_similarity > 0.5:
                    similarity_graph.add_edge(i, j, weight=cosine_similarity)
        
        # Apply PageRank
        pagerank = nx.pagerank(similarity_graph)
        return pagerank
    
    def cluster_based_summarization(self, sentences, sentence_vectors, num_clusters=3):
        """
        Cluster-based summarization to capture diverse content
        """
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(sentence_vectors)
        
        # Find representative sentences from each cluster
        cluster_representatives = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_sentences = [sentences[i] for i in cluster_indices]
            
            # Use TF-IDF to find most representative sentence
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(cluster_sentences)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Select top sentence from cluster
            top_sentence_index = cluster_indices[np.argmax(sentence_scores)]
            cluster_representatives.append(sentences[top_sentence_index])
        
        return cluster_representatives
    
    def generate_summary(self, text, summary_length=3):
        """
        Comprehensive summarization pipeline
        """
        # Preprocess text
        processed_sentences = self.preprocess_text(text)
        
        # Compute semantic vectors
        sentence_vectors = self.semantic_similarity(processed_sentences)
        
        # Multiple summarization techniques
        pagerank_scores = self.advanced_textrank(processed_sentences, sentence_vectors)
        cluster_summary = self.cluster_based_summarization(processed_sentences, sentence_vectors)
        
        # Combine approaches
        # Top sentences by PageRank
        top_pagerank_sentences = sorted(
            pagerank_scores, 
            key=pagerank_scores.get, 
            reverse=True
        )[:summary_length]
        
        # Final summary selection
        final_summary_indices = list(set(top_pagerank_sentences + 
            [processed_sentences.index(sent) for sent in cluster_summary]))
        
        final_summary = [processed_sentences[idx] for idx in sorted(final_summary_indices)]
        
        return final_summary[:summary_length]

# Example usage
def demonstrate_advanced_summarization():
    sample_text = """
    Artificial Intelligence (AI) has rapidly evolved over the past decade, transforming 
    multiple sectors of human activity. From healthcare to finance, AI technologies are 
    reshaping how we approach complex problems. Machine learning algorithms can now 
    detect diseases with unprecedented accuracy, predict financial market trends, and 
    even assist in scientific research by processing vast amounts of data far beyond 
    human capabilities. However, this technological revolution also brings significant 
    ethical challenges. Questions about data privacy, algorithmic bias, and the potential 
    displacement of human workers are at the forefront of ongoing debates. Despite these 
    concerns, the potential of AI to solve global challenges like climate change, 
    medical research, and resource optimization remains immense. Researchers are 
    continuously developing more sophisticated neural networks and exploring concepts 
    like explainable AI to make these systems more transparent and trustworthy.
    """
    
    summarizer = AdvancedSummarizer()
    summary = summarizer.generate_summary(sample_text)
    
    print("Advanced Summary:")
    for sent in summary:
        print("- " + sent)

if __name__ == "__main__":
    demonstrate_advanced_summarization()
```

## Key Components of Advanced Summarization

### 1. Semantic Understanding
- **Word2Vec Embedding**: Captures semantic relationships between words
- Converts text into dense vector representations
- Allows understanding of context beyond simple word frequency

### 2. Multi-Stage Processing
- **Preprocessing**:
  - Lemmatization
  - Named Entity Recognition
  - Stopword Removal
- **Semantic Similarity Calculation**
- **Graph-Based Ranking**
- **Clustering**

### 3. Advanced Techniques Integrated
- **TextRank Algorithm**
- **K-Means Clustering**
- **TF-IDF Weighting**
- **Semantic Vector Representation**

## Advantages Over Simple Extractive Methods

1. **Contextual Understanding**
   - Captures semantic meaning
   - Considers word relationships
   - Identifies important concepts

2. **Diverse Content Selection**
   - Uses clustering to ensure summary covers different aspects
   - Prevents repetitive or redundant information

3. **Adaptive Summarization**
   - Works across different document types
   - Adjustable summary length
   - Handles complex, multi-topic texts

## Required Dependencies
- spaCy
- NLTK
- Gensim
- scikit-learn
- NumPy

## Limitations Compared to LLMs
- Cannot generate novel sentences
- Limited to extracting existing content
- No true understanding of deep semantics
- Cannot handle extremely complex or ambiguous texts

## Potential Improvements
1. Incorporate domain-specific word embeddings
2. Add more advanced clustering techniques
3. Integrate sentiment analysis
4. Develop more sophisticated semantic similarity metrics

## When to Use
- Technical documents
- Research papers
- Long-form articles
- Comprehensive reports
- Scenarios with structured, factual content

## Performance Considerations
- Computational complexity increases with text length
- Requires pre-trained language models
- Memory-intensive for very large texts

---

# DEPLOYMENT ARCHITECTURE & HOSTING GUIDE

## Project Goal
Build a **separate microservice** that provides text summarization via REST API for the LLM-speech browser extension. This service will run Python code with heavy NLP dependencies that cannot run directly in the browser.

## Why Separate Service?
- **Browser extensions run JavaScript only** - cannot execute Python
- **Heavy dependencies** (spaCy ~500MB, sklearn, gensim) cannot run in browser
- **Clean separation** - extension stays lightweight, service scales independently
- **Reusable** - other projects can use the same API

---

## Architecture Options

### Option 1: API Only (Minimal)
```
Chrome Extension (JS) → REST API → Python Summarizer → JSON Response
```

**Pros**: Simple, minimal code
**Cons**: No way to test/demo without extension

### Option 2: API + Web Platform (RECOMMENDED)
```
Chrome Extension → REST API ┐
                            ├→ Python Summarizer
Web Dashboard   → Same API  ┘
```

**Components:**
- **Backend**: Flask/FastAPI REST API
- **Frontend**: Simple HTML/CSS/JS demo page
- **Shared Logic**: AdvancedSummarizer class

**Benefits:**
- Test summarizer without loading extension
- Demo page for GitHub README
- Easy debugging and development
- Show off your work publicly

---

## Repository Structure (Recommended)

```
llm-speech-summarizer-api/
├── app.py                    # Flask/FastAPI main server
├── summarizer.py             # AdvancedSummarizer class (from this doc)
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version (optional)
├── .env.example              # Environment variable template
├── templates/
│   └── index.html           # Web demo interface
├── static/
│   ├── style.css            # Demo page styles
│   └── script.js            # Demo page logic
├── tests/
│   ├── test_summarizer.py   # Unit tests
│   └── test_api.py          # API endpoint tests
├── .gitignore
├── README.md                 # Setup and API documentation
└── docs/
    └── API.md               # Detailed API specification
```

---

## API Specification

### Endpoint: POST /summarize

**Request:**
```json
{
  "text": "Long text to be summarized...",
  "summary_length": 3,
  "min_text_length": 500
}
```

**Parameters:**
- `text` (string, required): Text to summarize
- `summary_length` (int, optional): Number of sentences in summary (default: 3)
- `min_text_length` (int, optional): Minimum character count to trigger summarization (default: 500)

**Response (Success):**
```json
{
  "status": "success",
  "original_length": 1250,
  "summary": [
    "First key sentence from the text.",
    "Second important sentence.",
    "Third relevant sentence."
  ],
  "summary_text": "First key sentence from the text. Second important sentence. Third relevant sentence.",
  "processing_time_ms": 1234
}
```

**Response (Text Too Short):**
```json
{
  "status": "skipped",
  "message": "Text too short for summarization",
  "original_length": 300,
  "original_text": "The original short text..."
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

### Endpoint: GET /health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

## Extension Integration

### How Extension Will Use the API

In `content.js`, when auto-speak is enabled and response is detected:

```javascript
async function summarizeIfNeeded(responseText) {
  // Only summarize if text is long
  if (responseText.length < 500) {
    return responseText; // Use original
  }

  try {
    const response = await fetch('https://your-api.onrender.com/summarize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: responseText,
        summary_length: 3,
        min_text_length: 500
      })
    });

    const data = await response.json();

    if (data.status === 'success') {
      return data.summary_text;
    } else {
      return responseText; // Fallback to original
    }
  } catch (error) {
    console.error('Summarization failed:', error);
    return responseText; // Fallback
  }
}

// In response detection:
if (autoSpeak && responseText.length > 500) {
  showNotification('Summarizing response...', 'info');
  const summaryText = await summarizeIfNeeded(responseText);
  const announcementText = 'Here is a summary of what Claude has come up with. ' + summaryText;
  speakText(announcementText);
} else if (autoSpeak) {
  const announcementText = 'Here is what Claude has come up with. ' + responseText;
  speakText(announcementText);
}
```

### Extension Settings (Future Enhancement)

Add to `popup.html`:
```html
<div class="setting-item">
  <div class="setting-info">
    <label for="useSummaryToggle">Smart summarization</label>
    <p class="setting-description">Summarize long responses before speaking (requires API)</p>
  </div>
  <label class="toggle">
    <input type="checkbox" id="useSummaryToggle">
    <span class="toggle-slider"></span>
  </label>
</div>

<div class="setting-item">
  <label>Summary API URL:</label>
  <input type="text" id="apiUrlInput" placeholder="https://your-api.onrender.com">
</div>
```

---

## Hosting Options & Costs

### Free Hosting (With Limitations)

#### 1. Render (RECOMMENDED FOR STARTING)
- **Free Tier**: 750 hours/month
- **Limitations**:
  - Spins down after 15 min inactivity
  - Cold start: 20-60 seconds for first request
  - 512 MB RAM (tight for spaCy models)
- **Deployment**: Connect GitHub, auto-deploy on push
- **Cost**: FREE forever
- **Upgrade**: $7/month for always-on, 512MB RAM
- **Best For**: Low-traffic, hobby projects

**Setup:**
1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: llm-speech-summarizer
    env: python
    buildCommand: "pip install -r requirements.txt && python -m spacy download en_core_web_sm"
    startCommand: "gunicorn app:app"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

#### 2. Google Cloud Run
- **Free Tier**: 2M requests/month, 360K GB-seconds
- **Limitations**:
  - Cold starts (but faster than Render)
  - 1 GB RAM default (can increase)
- **Cost**: FREE for low traffic, ~$5-10/month for moderate use
- **Best For**: Production-ready free tier

**Setup:**
```bash
gcloud run deploy llm-summarizer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi
```

#### 3. AWS Lambda + API Gateway
- **Free Tier**: 1M requests/month forever
- **Limitations**:
  - Complex setup
  - 10 GB memory max (good for spaCy)
  - 15 min timeout
- **Cost**: FREE for <1M requests, then $0.20 per 1M
- **Best For**: Already familiar with AWS

#### 4. Fly.io
- **Free Tier**: 3 shared-cpu VMs, 160GB bandwidth
- **Cost**: FREE for small apps
- **Best For**: Good balance of free resources

#### 5. Railway
- **Free Tier**: $5 credit/month
- **Limitations**: Credit runs out, then need paid plan
- **Cost**: Usage-based after credit
- **Best For**: Testing, not long-term free

### Paid Hosting (For Production)

#### Render (Always-On)
- **Cost**: $7/month (512 MB RAM) to $25/month (2 GB RAM)
- **Pros**: Simple, automatic deployments
- **Best For**: Small to medium traffic

#### Heroku
- **Cost**: $7/month (Eco) to $25/month (Basic)
- **Pros**: Easy setup, lots of add-ons
- **Cons**: No free tier anymore

#### DigitalOcean App Platform
- **Cost**: $5/month basic, $12/month for better specs
- **Pros**: Good performance, simple

---

## Cold Start Mitigation

### Problem:
Free tiers sleep after inactivity. First request takes 20-60 seconds.

### Solutions:

1. **Keep-Alive Ping** (use carefully, some services prohibit)
```python
# Use external cron service (cron-job.org, UptimeRobot)
# Ping /health every 14 minutes
```

2. **User Experience**
```javascript
// In extension
showNotification('Waking up summarizer (first use may take 30s)...', 'info');
```

3. **Upgrade to Paid** ($7/month for always-on)

---

## Implementation Steps

### Phase 1: Core API (Week 1)
1. Create new GitHub repo: `llm-speech-summarizer-api`
2. Copy `AdvancedSummarizer` class to `summarizer.py`
3. Create Flask app with `/summarize` endpoint in `app.py`
4. Add error handling and validation
5. Create `requirements.txt`
6. Test locally: `python app.py`

### Phase 2: Web Demo (Week 1-2)
1. Create `templates/index.html` with textarea and button
2. Add API call from frontend
3. Display summary results
4. Style with simple CSS

### Phase 3: Deploy to Render (Week 2)
1. Push to GitHub
2. Connect Render to repo
3. Configure build and start commands
4. Deploy and get public URL
5. Test API endpoint

### Phase 4: Extension Integration (Week 2)
1. Add summarization feature toggle to extension
2. Store API URL in chrome.storage
3. Implement `summarizeIfNeeded()` function
4. Add loading states and error handling
5. Test end-to-end flow

### Phase 5: Optimization (Week 3+)
1. Add caching for repeated texts
2. Optimize model loading (load once at startup)
3. Add request logging and analytics
4. Implement rate limiting
5. Add API key authentication (optional)

---

## Flask App Example (app.py)

```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from summarizer import AdvancedSummarizer
import time

app = Flask(__name__)
CORS(app)  # Allow extension to call API

# Load model once at startup (not per request)
print("Loading spaCy model...")
summarizer = AdvancedSummarizer()
print("Model loaded successfully!")

@app.route('/')
def home():
    """Serve web demo page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'model_loaded': True
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Main summarization endpoint"""
    try:
        # Parse request
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: text',
                'error_type': 'ValidationError'
            }), 400

        text = data['text']
        summary_length = data.get('summary_length', 3)
        min_text_length = data.get('min_text_length', 500)

        # Validate
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Text must be a non-empty string',
                'error_type': 'ValidationError'
            }), 400

        # Check if text is too short
        if len(text) < min_text_length:
            return jsonify({
                'status': 'skipped',
                'message': 'Text too short for summarization',
                'original_length': len(text),
                'original_text': text
            })

        # Summarize
        start_time = time.time()
        summary_sentences = summarizer.generate_summary(text, summary_length)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        summary_text = ' '.join(summary_sentences)

        return jsonify({
            'status': 'success',
            'original_length': len(text),
            'summary': summary_sentences,
            'summary_text': summary_text,
            'processing_time_ms': round(processing_time, 2)
        })

    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## requirements.txt

```txt
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
spacy==3.7.2
nltk==3.8.1
gensim==4.3.2
scikit-learn==1.3.2
numpy==1.26.2
networkx==3.2.1
```

---

## Testing Strategy

### Local Testing
```bash
# Terminal 1: Run API
python app.py

# Terminal 2: Test with curl
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "summary_length": 3
  }'
```

### Extension Testing
1. Update extension to point to `http://localhost:5000`
2. Test summarization flow
3. Once working, update to production URL

---

## Security Considerations

### For Public API:
1. **Rate Limiting**: Prevent abuse
```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])
```

2. **API Key (Optional)**:
```python
API_KEY = os.environ.get('API_KEY')

@app.before_request
def check_api_key():
    if request.endpoint == 'summarize':
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
```

3. **Input Validation**: Already implemented in example above

4. **CORS**: Already using flask-cors to allow extension access

---

## Monitoring & Maintenance

### Log Requests:
```python
import logging
logging.basicConfig(filename='api.log', level=logging.INFO)

@app.route('/summarize', methods=['POST'])
def summarize():
    logging.info(f"Request from {request.remote_addr}, text_length={len(text)}")
    # ... rest of code
```

### Track Usage:
- Monitor request count on Render dashboard
- Set up alerts for high error rates
- Check cold start frequency

---

## Cost Projections

### Low Traffic (< 10K requests/month):
- **Free tier sufficient** (Render, Cloud Run, Lambda)
- Cost: $0/month

### Medium Traffic (10K - 100K requests/month):
- **Render**: $7/month (always-on)
- **Cloud Run**: $5-15/month (usage-based)
- **AWS Lambda**: $2-5/month

### High Traffic (>100K requests/month):
- **Cloud Run**: $15-30/month
- **Dedicated Server**: $12-25/month (DigitalOcean)
- Consider caching and optimization

---

## Success Criteria

### Phase 1: MVP
- [ ] API deployed and accessible
- [ ] `/summarize` endpoint working
- [ ] Web demo functional
- [ ] Extension can call API successfully

### Phase 2: Production Ready
- [ ] Error handling robust
- [ ] Cold starts acceptable (<60s)
- [ ] Documentation complete
- [ ] Tests passing

### Phase 3: Optimized
- [ ] Processing time <2s for typical text
- [ ] Rate limiting implemented
- [ ] Monitoring set up
- [ ] Caching for common requests

---

## Fallback Strategy

If API is down or slow, extension should:
1. Show user notification: "Summarizer unavailable, using full response"
2. Fall back to speaking full text
3. Log error for debugging

```javascript
async function summarizeIfNeeded(responseText) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: responseText }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    const data = await response.json();
    return data.status === 'success' ? data.summary_text : responseText;

  } catch (error) {
    console.error('Summarization failed:', error);
    showNotification('Summarizer unavailable, using full response', 'warning');
    return responseText; // Graceful degradation
  }
}
```

---

## Next Steps

1. **Create new repo**: `llm-speech-summarizer-api`
2. **Copy this doc** to new repo as implementation guide
3. **Start with Phase 1**: Build core API locally
4. **Test thoroughly** before deploying
5. **Deploy to Render free tier**
6. **Integrate with extension** once API is stable
7. **Monitor and optimize** based on usage

---

## Questions & Design Decisions

### Q: Should summarization be always-on or toggle?
**A**: Make it a **toggle** in extension settings. Let users choose.

### Q: What if summarization takes too long?
**A**: Show loading notification, implement 10s timeout, fall back to full text.

### Q: Cache summaries?
**A**: Yes, for Phase 3. Hash input text, cache result for 1 hour.

### Q: API authentication needed?
**A**: Not initially. Add if abuse becomes problem.

### Q: Support other languages?
**A**: Start English only. Add multilingual models later if needed.

---

## References & Resources

- Flask Documentation: https://flask.palletsprojects.com/
- Render Deploy Guide: https://render.com/docs/deploy-flask
- spaCy Models: https://spacy.io/models
- CORS for Extensions: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS

---

**Document Version**: 1.0
**Last Updated**: 2025-11-21
**Maintained By**: LLM-speech project team
