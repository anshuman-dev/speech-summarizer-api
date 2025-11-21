"""
Speech Summarizer API - Flask Application
REST API for text summarization using Sentence-BERT
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from summarizer import ProductionSummarizer
import time
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for browser extension access
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, specify actual origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model once at startup (not per request)
# This improves performance and reduces memory usage
logger.info("=" * 60)
logger.info("Starting Speech Summarizer API")
logger.info(f"Python: {sys.version.split()[0]}")
logger.info(f"Working Directory: {os.getcwd()}")
logger.info("=" * 60)

try:
    logger.info("Loading Sentence-BERT model (this may take 30-60 seconds)...")
    import_start = time.time()
    summarizer = ProductionSummarizer()
    model_info = summarizer.get_model_info()
    import_time = time.time() - import_start
    logger.info(f"✓ Summarizer loaded successfully in {import_time:.2f}s")
    logger.info(f"  Model: {model_info['model_name']}")
    logger.info(f"  Type: {model_info['type']}")
    logger.info(f"  Embedding Dimension: {model_info['embedding_dimension']}")
    logger.info("=" * 60)
    logger.info("Flask application ready to accept requests")
    logger.info("=" * 60)
except Exception as e:
    logger.error(f"✗ Failed to load summarizer: {str(e)}", exc_info=True)
    raise


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    """Serve web demo page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for monitoring and uptime checks

    Returns:
        JSON with status, version, and model information
    """
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'model': model_info['model_name'],
        'model_loaded': True,
        'embedding_dimension': model_info['embedding_dimension']
    })


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed model information

    Returns:
        JSON with model specifications
    """
    return jsonify(model_info)


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Main summarization endpoint

    Request JSON:
        {
            "text": "Long text to summarize...",
            "summary_length": 3,  // optional, default: 3
            "min_text_length": 500  // optional, default: 500
        }

    Response JSON (Success):
        {
            "status": "success",
            "original_length": 1250,
            "summary_text": "Summary goes here...",
            "processing_time_ms": 234.56
        }

    Response JSON (Skipped - text too short):
        {
            "status": "skipped",
            "message": "Text too short for summarization",
            "original_length": 300,
            "original_text": "..."
        }

    Response JSON (Error):
        {
            "status": "error",
            "message": "Error description",
            "error_type": "ValidationError"
        }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body must be JSON',
                'error_type': 'ValidationError'
            }), 400

        if 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: text',
                'error_type': 'ValidationError'
            }), 400

        text = data['text']
        summary_length = data.get('summary_length', 3)
        min_text_length = data.get('min_text_length', 500)

        # Validate text
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Text must be a non-empty string',
                'error_type': 'ValidationError'
            }), 400

        # Validate summary_length
        if not isinstance(summary_length, int) or summary_length < 1:
            return jsonify({
                'status': 'error',
                'message': 'summary_length must be a positive integer',
                'error_type': 'ValidationError'
            }), 400

        if summary_length > 10:
            return jsonify({
                'status': 'error',
                'message': 'summary_length cannot exceed 10 sentences',
                'error_type': 'ValidationError'
            }), 400

        # Check if text is too short for summarization
        if len(text) < min_text_length:
            logger.info(f"Text too short ({len(text)} chars), skipping summarization")
            return jsonify({
                'status': 'skipped',
                'message': 'Text too short for summarization',
                'original_length': len(text),
                'original_text': text
            })

        # Log request
        logger.info(f"Summarization request: {len(text)} chars, {summary_length} sentences")

        # Generate summary
        start_time = time.time()
        summary_text = summarizer.generate_summary(text, summary_length)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Summary generated in {processing_time:.2f}ms")

        return jsonify({
            'status': 'success',
            'original_length': len(text),
            'summary_length': len(summary_text),
            'summary_text': summary_text,
            'processing_time_ms': round(processing_time, 2),
            'compression_ratio': round(len(summary_text) / len(text), 3)
        })

    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'error_type': 'NotFound'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error_type': 'InternalServerError'
    }), 500


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Debug mode: {debug}")

    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )
