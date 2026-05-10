// =============================================================
// PneumoScan AI — Main Application Logic
// =============================================================

const APP = {
    model: null,
    isModelLoaded: false,
    isDemoMode: true,
    classNames: ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia'],
    imgSize: 224,
};

// ---- DOM Elements ----
const $ = (sel) => document.querySelector(sel);
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');
const uploadSection = $('#upload-section');
const analysisSection = $('#analysis-section');
const analyzingState = $('#analyzing-state');
const resultsContent = $('#results-content');
const scanOverlay = $('#scan-overlay');
const demoBanner = $('#demo-banner');
const modelStatusDot = $('#model-status-dot');
const modelStatusText = $('#model-status-text');
const btnNewScan = $('#btn-new-scan');

// ---- Modal Elements ----
const errorModal = $('#error-modal');
const btnModalCancel = $('#btn-modal-cancel');

// ---- Initialize ----
document.addEventListener('DOMContentLoaded', () => {
    setupUpload();
    setupDragDrop();
    loadModel();
    setupModal();
    btnNewScan.addEventListener('click', resetToUpload);
});

// ---- Model Loading ----
async function loadModel() {
    try {
        modelStatusText.textContent = 'Loading Model...';
        modelStatusDot.className = 'status-dot';

        // Try to load TensorFlow.js model from /model/ directory
        APP.model = await tf.loadLayersModel('./model/model.json');
        APP.isModelLoaded = true;
        APP.isDemoMode = false;

        modelStatusDot.classList.add('ready');
        modelStatusText.textContent = 'Model Ready';
        demoBanner.classList.remove('visible');

        console.log('✅ Model loaded successfully!');
    } catch (err) {
        console.warn('⚠️ No model found, running in Demo Mode:', err.message);
        APP.isDemoMode = true;

        modelStatusDot.classList.add('ready');
        modelStatusText.textContent = 'Demo Mode';
        
        // Update demo banner to explain WHY it's in demo mode
        const demoTitle = demoBanner.querySelector('strong');
        const demoText = demoBanner.querySelector('p');
        
        if (err.message.toLowerCase().includes('fetch')) {
            demoTitle.textContent = 'Security Block Detected';
            demoText.innerHTML = 'Browser security prevents loading the model from a local file. <br><strong>Fix:</strong> Right-click <code>index.html</code> in VS Code and select <strong>"Open with Live Server"</strong>.';
        } else {
            demoTitle.textContent = 'Model Load Failed';
            demoText.textContent = err.message;
        }
        
        demoBanner.classList.add('visible');
    }
}

// ---- File Upload ----
function setupUpload() {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function setupDragDrop() {
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

// ---- Handle Uploaded File ----
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG, PNG).');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const imageData = e.target.result;
        
        // Create an image element to check features
        const img = new Image();
        img.onload = () => {
            if (isLikelyXray(img)) {
                showAnalysis(imageData, file);
            } else {
                showErrorModal();
            }
        };
        img.onerror = () => {
            alert('The image could not be loaded. It might be corrupted or in an unsupported format.');
            resetToUpload();
        };
        img.src = imageData;
    };
    reader.readAsDataURL(file);
}

// ---- X-Ray Validation Logic ----
function isLikelyXray(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Scale down for analysis
    canvas.width = 50;
    canvas.height = 50;
    ctx.drawImage(img, 0, 0, 50, 50);
    
    const imageData = ctx.getImageData(0, 0, 50, 50);
    const data = imageData.data;
    
    let totalVariance = 0;
    
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i+1];
        const b = data[i+2];
        
        // Calculate standard deviation between R, G, and B
        const avg = (r + g + b) / 3;
        const variance = Math.sqrt(
            (Math.pow(r - avg, 2) + Math.pow(g - avg, 2) + Math.pow(b - avg, 2)) / 3
        );
        totalVariance += variance;
    }
    
    const avgVariance = totalVariance / (data.length / 4);
    
    // X-rays are grayscale, so variance between RGB channels should be very low.
    // colorful images usually have avgVariance > 15-20.
    // Threshold of 12 is a safe middle ground.
    console.log('🔍 Image Color Variance:', avgVariance.toFixed(2));
    return avgVariance < 12;
}

// ---- Modal Logic ----

function setupModal() {
    btnModalCancel.addEventListener('click', () => {
        errorModal.classList.add('hidden');
        resetToUpload();
    });
}

function showErrorModal() {
    errorModal.classList.remove('hidden');
}

// ---- Show Analysis View ----
function showAnalysis(imageData, file) {
    // Switch views
    uploadSection.classList.add('hidden');
    analysisSection.classList.remove('hidden');

    // Show image
    const preview = $('#xray-preview');
    preview.src = imageData;

    // Image info
    $('#img-name').textContent = file.name;
    $('#img-size').textContent = formatFileSize(file.size);

    // Start analysis
    startAnalysis(imageData);
}

// ---- Start Analysis ----
async function startAnalysis(imageData) {
    // Show analyzing state
    analyzingState.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    scanOverlay.classList.add('active');
    $('#analysis-time').textContent = 'Analyzing...';

    const startTime = performance.now();

    let predictions;
    if (APP.isDemoMode) {
        // Simulate analysis with random delay
        await delay(1500 + Math.random() * 1000);
        predictions = generateDemoPrediction();
    } else {
        // Real model inference
        predictions = await runInference(imageData);
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    // Show results
    scanOverlay.classList.remove('active');
    analyzingState.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    $('#analysis-time').textContent = `${elapsed}s`;

    displayResults(predictions);
}

// ---- Real Model Inference ----
async function runInference(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            // Preprocess: resize to 224x224, normalize to [0, 1]
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([APP.imgSize, APP.imgSize])
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims(0);

            // Predict
            const output = APP.model.predict(tensor);
            const probs = output.dataSync();

            tensor.dispose();
            output.dispose();

            resolve({
                normal: probs[0],
                bacterial: probs[1],
                viral: probs[2]
            });
        };
        img.src = imageData;
    });
}

// ---- Demo Prediction ----
function generateDemoPrediction() {
    // Generate realistic-looking demo predictions
    const scenarios = [
        { normal: 0.92, bacterial: 0.05, viral: 0.03 },
        { normal: 0.08, bacterial: 0.78, viral: 0.14 },
        { normal: 0.05, bacterial: 0.12, viral: 0.83 },
        { normal: 0.85, bacterial: 0.10, viral: 0.05 },
        { normal: 0.03, bacterial: 0.88, viral: 0.09 },
        { normal: 0.11, bacterial: 0.07, viral: 0.82 },
    ];

    // Add slight randomness
    const base = scenarios[Math.floor(Math.random() * scenarios.length)];
    const noise = () => (Math.random() - 0.5) * 0.08;

    let normal = Math.max(0.01, base.normal + noise());
    let bacterial = Math.max(0.01, base.bacterial + noise());
    let viral = Math.max(0.01, base.viral + noise());

    // Normalize to sum to 1
    const total = normal + bacterial + viral;
    return {
        normal: normal / total,
        bacterial: bacterial / total,
        viral: viral / total
    };
}

// ---- Display Results ----
function displayResults(predictions) {
    const { normal, bacterial, viral } = predictions;

    // Find the highest prediction
    const maxVal = Math.max(normal, bacterial, viral);
    let diagnosis, icon, cardClass;

    if (maxVal === normal) {
        diagnosis = 'Normal';
        icon = '✅';
        cardClass = 'normal';
    } else if (maxVal === bacterial) {
        diagnosis = 'Bacterial Pneumonia';
        icon = '⚠️';
        cardClass = 'pneumonia';
    } else {
        diagnosis = 'Viral Pneumonia';
        icon = '⚠️';
        cardClass = 'pneumonia';
    }

    // Update diagnosis card
    const diagCard = $('#diagnosis-card');
    diagCard.className = 'diagnosis-card ' + cardClass;
    $('#diagnosis-icon').textContent = icon;
    $('#diagnosis-value').textContent = diagnosis;
    $('#confidence-badge').textContent = (maxVal * 100).toFixed(1) + '%';

    // Animate probability bars
    setTimeout(() => {
        $('#prob-normal').style.width = (normal * 100) + '%';
        $('#prob-bacterial').style.width = (bacterial * 100) + '%';
        $('#prob-viral').style.width = (viral * 100) + '%';

        $('#prob-normal-val').textContent = (normal * 100).toFixed(1) + '%';
        $('#prob-bacterial-val').textContent = (bacterial * 100).toFixed(1) + '%';
        $('#prob-viral-val').textContent = (viral * 100).toFixed(1) + '%';
    }, 100);
}

// ---- Reset to Upload View ----
function resetToUpload() {
    analysisSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    fileInput.value = '';

    // Reset probability bars
    $('#prob-normal').style.width = '0%';
    $('#prob-bacterial').style.width = '0%';
    $('#prob-viral').style.width = '0%';
}

// ---- Utilities ----
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}
