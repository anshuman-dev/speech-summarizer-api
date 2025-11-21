// Speech Summarizer API - Frontend JavaScript

// DOM Elements
const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');
const sentenceCount = document.getElementById('sentenceCount');
const summaryLength = document.getElementById('summaryLength');
const summaryLengthValue = document.getElementById('summaryLengthValue');
const minTextLength = document.getElementById('minTextLength');
const minTextLengthValue = document.getElementById('minTextLengthValue');
const summarizeBtn = document.getElementById('summarizeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingState = document.getElementById('loadingState');
const outputSection = document.getElementById('outputSection');
const errorSection = document.getElementById('errorSection');
const summaryText = document.getElementById('summaryText');
const originalLength = document.getElementById('originalLength');
const summaryLengthSpan = document.getElementById('summaryLength');
const compressionRatio = document.getElementById('compressionRatio');
const processingTime = document.getElementById('processingTime');
const errorMessage = document.getElementById('errorMessage');
const copyBtn = document.getElementById('copyBtn');
const modelDetails = document.getElementById('modelDetails');

// API Base URL (change for production)
const API_BASE_URL = window.location.origin;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateTextInfo();
    loadModelInfo();

    // Sample text for testing
    const sampleText = ``;

    // Set sample text if textarea is empty
    if (!inputText.value.trim()) {
        inputText.value = sampleText;
        updateTextInfo();
    }
});

// Event Listeners
inputText.addEventListener('input', updateTextInfo);

summaryLength.addEventListener('input', (e) => {
    summaryLengthValue.textContent = e.target.value;
});

minTextLength.addEventListener('input', (e) => {
    minTextLengthValue.textContent = e.target.value;
});

summarizeBtn.addEventListener('click', handleSummarize);
clearBtn.addEventListener('click', handleClear);
copyBtn.addEventListener('click', handleCopy);

// Enter key in textarea (Ctrl/Cmd + Enter to summarize)
inputText.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        handleSummarize();
    }
});

// Functions
function updateTextInfo() {
    const text = inputText.value;
    const chars = text.length;

    // Count sentences (rough estimate)
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;

    charCount.textContent = `${chars} characters`;
    sentenceCount.textContent = `~${sentences} sentences`;

    // Update button state
    const minLength = parseInt(minTextLength.value);
    if (chars < minLength) {
        summarizeBtn.disabled = true;
        summarizeBtn.textContent = `Need ${minLength - chars} more characters`;
    } else {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = 'Generate Summary';
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        const data = await response.json();

        modelDetails.textContent = `Model: ${data.model_name} | Embedding Dim: ${data.embedding_dimension} | License: ${data.license}`;
    } catch (error) {
        console.error('Failed to load model info:', error);
        modelDetails.textContent = 'Model info unavailable';
    }
}

async function handleSummarize() {
    const text = inputText.value.trim();

    if (!text) {
        showError('Please enter some text to summarize');
        return;
    }

    // Hide previous results
    hideError();
    outputSection.style.display = 'none';

    // Show loading state
    loadingState.style.display = 'block';
    summarizeBtn.disabled = true;

    try {
        const requestBody = {
            text: text,
            summary_length: parseInt(summaryLength.value),
            min_text_length: parseInt(minTextLength.value)
        };

        const response = await fetch(`${API_BASE_URL}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        // Hide loading
        loadingState.style.display = 'none';

        if (response.ok) {
            if (data.status === 'success') {
                displaySummary(data);
            } else if (data.status === 'skipped') {
                showError(`${data.message} (${data.original_length} characters). Minimum required: ${requestBody.min_text_length} characters.`);
            } else {
                showError(data.message || 'Unknown error occurred');
            }
        } else {
            showError(data.message || `Server error: ${response.status}`);
        }

    } catch (error) {
        loadingState.style.display = 'none';
        showError(`Network error: ${error.message}. Make sure the API server is running.`);
        console.error('Summarization error:', error);
    } finally {
        summarizeBtn.disabled = false;
        updateTextInfo(); // Reset button text
    }
}

function displaySummary(data) {
    // Display summary text
    summaryText.textContent = data.summary_text;

    // Display statistics
    originalLength.textContent = `${data.original_length} chars`;
    summaryLengthSpan.textContent = `${data.summary_length} chars`;
    compressionRatio.textContent = `${(data.compression_ratio * 100).toFixed(1)}%`;
    processingTime.textContent = `${data.processing_time_ms}ms`;

    // Show output section
    outputSection.style.display = 'block';

    // Smooth scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

function handleClear() {
    inputText.value = '';
    outputSection.style.display = 'none';
    hideError();
    updateTextInfo();
    inputText.focus();
}

async function handleCopy() {
    const text = summaryText.textContent;

    try {
        await navigator.clipboard.writeText(text);

        // Visual feedback
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✓ Copied!';
        copyBtn.style.backgroundColor = '#10b981';
        copyBtn.style.color = 'white';

        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.backgroundColor = '';
            copyBtn.style.color = '';
        }, 2000);

    } catch (error) {
        console.error('Failed to copy:', error);
        alert('Failed to copy to clipboard');
    }
}

// Utility function for API testing (accessible from console)
window.testAPI = async function() {
    console.log('Testing API endpoints...');

    // Test health
    try {
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        const healthData = await healthResponse.json();
        console.log('✓ Health check:', healthData);
    } catch (error) {
        console.error('✗ Health check failed:', error);
    }

    // Test model info
    try {
        const modelResponse = await fetch(`${API_BASE_URL}/model-info`);
        const modelData = await modelResponse.json();
        console.log('✓ Model info:', modelData);
    } catch (error) {
        console.error('✗ Model info failed:', error);
    }

    console.log('API test complete. Check results above.');
};

// Log ready message
console.log('Speech Summarizer API Frontend Ready');
console.log('Run testAPI() to test the API endpoints');
