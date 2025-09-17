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

**Action Shot Extractor is completely free** with AI-powered detection, no subscriptions or watermarks. Records whole match at 4K/60fps, uses computer vision to find frames where ball and target player appear together.

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

Result: Curated high-action shots without manual scrubbing.

## Technical Stack & Features

- **Python 3.13+** with Poetry, works on Windows/macOS/Linux
- **YOLOv8 + SegFormer** – AI models for object detection and scene understanding
- **Dual Detection** – Color-based OR reference frame matching for player identification
- **Parallel Processing** – Multi-threaded analysis for faster processing on multi-core systems
- **CLI + Web Interface** – `action-shot-extractor` command-line tool + Streamlit web UI
- **4K/60fps Support** – High-resolution video processing with sharpness filtering
- **Lazy Loading** – Models load on-demand to reduce startup time

## Installation

**Docker (Recommended):**
```bash
docker-compose up --build
# Open http://localhost:8501 for web interface
```

**Local:**
```bash
pip install poetry
poetry install
# Or: pip install -r requirements.txt
```

## Usage

### Web Interface
Open http://localhost:8501 in browser → Drag & drop video → Pick player color → Extract frames

### Command Line
```bash
# Color detection
action-shot-extractor --video match.mp4 --hex "#E10600" --workers 4

# Reference frames
action-shot-extractor --video match.mp4 --reference-frames ./player_refs/

# Help
action-shot-extractor --help
```

## AI Models Used

Action Shot Extractor leverages two pre-trained AI models:

- **YOLOv8s** - Object detection for identifying people and sports balls
- **SegFormer** - Semantic segmentation from Hugging Face for scene understanding

## Development

```bash
poetry run pytest  # Run tests in Poetry environment
```
