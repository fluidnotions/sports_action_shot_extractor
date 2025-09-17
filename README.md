# Action Shot Extractor

Capturing the perfect action shot from a soccer game can feel impossible. Do you click too early? Too late? Or end up with dozens of blurry frames? Instead of guessing, **Action Shot Extractor** automates the process of finding the exact moments worth keeping.

## 💰 Why This Tool Exists (Free vs Paid Alternatives)

Most AI sports video analysis tools in 2025 are expensive subscription services:

**Paid Solutions ($50-500+/month):**
- **Veo** - Professional camera system + AI analysis (subscription + hardware costs)
- **Pixellot** - Multi-camera automated production (enterprise pricing)
- **SportsVisio** - AI stats and highlights (monthly subscription)
- **Magnifi** - Cloud-based highlight generation (usage-based pricing)
- **PlaySight** - Smart court technology (venue installation costs)

**Free Alternatives (Limited):**
- **Manual frame extraction** (VLC, online tools) - No AI, very time-consuming
- **Basic video editors** (iMovie, etc.) - Manual highlight creation only
- **Trial versions** - Limited features, watermarks, time restrictions

**Action Shot Extractor is completely free** and provides AI-powered automatic detection without subscriptions, usage limits, or watermarks. Perfect for parents, amateur athletes, and coaches who want professional results without the professional price tag.

This tool takes a different approach: record the whole match in high-quality video (4K at 60fps) and let the software do the heavy lifting. Using computer vision, the app identifies frames where the ball and the **target player** appear together.

## Player Identification Methods

Choose the method that works best for your situation:

### Method 1: Color-Based Detection (Recommended for unique colors)
Perfect when your target player wears distinctly colored gear:
- **Unique jersey colors** - Player in red while team wears blue
- **Distinctive accessories** - Bright colored shoes, socks, headbands, gloves
- **Goalkeeper gear** - Different colored jersey from field players

### Method 2: Reference Frame Matching (Recommended for same-colored teams)
Ideal when color alone isn't enough to distinguish players:
- Provide **3 reference images** of your target player (front view, side view, back view)
- Uses advanced **ORB feature detection** for robust matching across different angles
- Works even when players wear identical uniforms

The result is a curated set of high-action shots without the pain of scrubbing through half an hour of footage manually.

## Key Features
- **Targeted Frame Extraction** – Finds frames where the player and the ball appear together, zeroing in on real action.
- **Dual Detection Modes** – Color-based detection for unique gear OR reference frame matching for identical uniforms.
- **Parallel Processing** – Multi-threaded frame analysis for faster processing on multi-core systems.
- **High-Resolution Support** – Works with 4K/60fps video for crystal-clear stills.
- **Efficient Workflow** – Replaces tedious manual scrubbing with an automated pass through the entire video.  

## Why It Matters
For parents, athletes, and casual photographers, this solves the pain of missing key shots or spending hours reviewing video. Instead of guesswork, you get a curated set of **action-ready frames** you can save, share, or print.

## Technical

The project targets Python 3.13 and is designed to work well on Windows, macOS and Linux
through a Poetry-driven workflow.

## Features

- CLI interface exposed as `framescout`
- Lazy-loading utilities for YOLOv8 and SegFormer models
- Color-aware filtering helpers for hue, saturation and value thresholds
- Sharpness scoring to eliminate blurry frames before detection
- Optional dry-run mode for quickly validating configuration without heavy computation

## Installation

### Option 1: Docker (Recommended for Quick Demo)

The easiest way to try Action Shot Extractor is with Docker:

```bash
# Clone the repository
git clone <repository-url>
cd action-shot-extractor

# Build and run with Docker Compose
docker-compose up --build
```

Then open your browser to **http://localhost:8501** for the web interface!

### Option 2: Local Installation

```bash
pip install poetry
poetry install
```

If you prefer plain `pip`, install the dependencies listed in
[`requirements.txt`](requirements.txt).

## Usage

### Web Interface (Easiest)

If you used Docker, simply open **http://localhost:8501** in your browser. The web interface provides:

- 📁 **Drag & drop video upload**
- 🎨 **Color picker for player detection**
- 📸 **Reference image upload for complex scenarios**
- ⚙️ **Real-time parameter adjustment**
- 📦 **One-click download of extracted frames**

### Command Line Interface

```bash
action-shot-extractor --video "C:\\clips\\match.mp4" --hex "#E10600" --out "C:\\clips\\hits"
```

Run `action-shot-extractor --help` to see all available options. The most common parameters are:

- `--video` *(required)* – path to the source video file
- `--hex` – team color in hexadecimal form (e.g. `#E10600`) for color-based detection
- `--reference-frames` – directory containing front.jpg, side.jpg, back.jpg for feature-based detection
- `--workers` – number of parallel workers (default: auto-detect, max 4)
- `--out` – optional destination folder for extracted frames
- `--dry-run` – validate your setup without downloading models or extracting frames

**Examples:**

Color-based detection:
```bash
action-shot-extractor --video match.mp4 --hex "#E10600" --workers 4
```

Reference frame detection:
```bash
action-shot-extractor --video match.mp4 --reference-frames ./player_refs/ --workers 2
```

## AI Models Used

Action Shot Extractor leverages two pre-trained AI models:

- **YOLOv8s** - Object detection for identifying people and sports balls
- **SegFormer** - Semantic segmentation from Hugging Face for scene understanding

## Performance & Parallel Processing

**Multi-threaded processing** for optimal performance on multi-core systems. Each frame requires:
- YOLOv8 inference (~50-100ms per frame)
- SegFormer segmentation (~100-200ms per frame)
- Player matching (~10-50ms per frame)

**Processing time examples:**
- **Single-threaded**: 10-second 60fps video (600 frames) = ~2-5 minutes
- **4-core parallel**: Same video = ~30-75 seconds (3-4x speedup)

Use `--workers N` to control parallelism or `--workers 1` for sequential processing.

## Future Enhancements

### Sport-Specific Detection
- **Tennis**: Detect rackets + ball + player for volleys and serves
- **Basketball**: Detect hoop + ball + player for shots
- **Baseball**: Detect bat + ball + player for hits
- **Hockey**: Detect stick + puck + player for shots

### Ball Customization
- Configurable ball colors for specialized equipment
- Custom ball size filters
- Multi-ball tracking for training scenarios

### Multi-Camera Analysis
- **Multiple video input processing** - Sync and analyze 4+ camera angles simultaneously
- **Field coverage optimization** - Strategic camera placement for complete field visibility
- **Multi-player tracking** - Track different players across multiple camera views
- **Zoom coordination** - Combine wide-angle coverage with telephoto action shots
- **Professional setup support** - Smartphone arrays with varying lens capabilities
- **Synchronized extraction** - Merge action shots from all camera angles

### Performance Optimizations
- ✅ **Parallel processing for multi-core systems** (implemented)
- GPU acceleration for model inference
- Frame sampling strategies (process every Nth frame)
- Model quantization for faster inference

### Docker & Deployment
- ✅ **Docker containerization** (implemented)
- ✅ **Streamlit web interface** (implemented)
- Cloud deployment (AWS, GCP, Azure)
- API endpoints for integration

## Development

Use `pytest` to run the lightweight smoke tests:

```bash
poetry run pytest
```

The tests exercise the color utilities, sharpness heuristics and the CLI dry-run pathway.
