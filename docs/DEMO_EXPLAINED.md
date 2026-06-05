# What Happens When You Upload a Video

---

## The big picture 

When you upload a video on the website, the system does **not** just look at one picture. It watches the **eyes over time** — how they move, blink, and look — and compares that pattern to what it learned from real and fake videos during training.

The final answer is either **REAL** or **FAKE**, with a **confidence** percentage.

---

## Step-by-step: what happens after you click “Analyse”

### 1. Your video is sent to the server

The browser sends the video file to the **API** (the backend hosted on a cloud). 

### 2. The video is opened frame by frame

The program reads the video like a normal video player. It does **not** use every single frame (that would be slow). It samples about **10 frames per second** — enough to catch blinks and eye movement.

### 3. A face detector finds the eyes (MediaPipe)

For each sampled frame, **MediaPipe Face Mesh** finds landmarks on the face — especially around the eyes.

From those points it:

- Crops a small **eye region** image (224×224 pixels)
- Calculates **EAR** (Eye Aspect Ratio) — a number that says how open the eyes are

**EAR in plain English:**  
When the eyes are wide open, EAR is higher. When the person blinks and the eyelids close, EAR drops. This is a standard way researchers measure blinks from video.

### 4. The video is split into “sequences”

All the eye crops and EAR values are grouped into chunks of **8 frames in a row**. Each chunk is one **sequence**.

**Why sequences?**  
Deepfakes often look okay in a single frame but behave strangely over time — for example, blinking too little, too much, or at odd moments. The model is built to look at **short clips**, not one photo.

**Example:**  
A 30-second video might give you 10–20 sequences, depending on length and how many frames had a detectable face.

### 5. The trained model scores each sequence

For **each** sequence, the **LRCN-ViT** model does two things:

1. **Vision part (ViT):** Looks at the 8 eye images and extracts visual features — texture, shape, small artifacts.
2. **Time part (LSTM):** Links those features across the 8 frames so it sees **motion over time**.
3. **Blink part:** Also uses the 8 EAR numbers so blink timing is part of the decision.

The model outputs a score between **0% and 100%** meaning “how fake does this short clip look?”  
That number is the **sequence score** (one score per sequence).

### 6. Everything is combined into the final result

- **Confidence** = average of all sequence scores  
- **REAL or FAKE** = if the average is **50% or higher** → FAKE, otherwise REAL  
- **Blink rate** = counted separately from the EAR values (see below)  
- **Chart** = each sequence score drawn as a line so you can see which parts of the video looked more suspicious

---

## What each number on the screen means

### REAL / FAKE (the big label)

This is the **final verdict**.

| Label | Meaning |
|-------|---------|
| **REAL** | The model thinks the video is probably authentic. |
| **FAKE** | The model thinks the video is probably a deepfake. |

**How it is decided:**  
Take the average of all sequence scores. If that average is **0.5 (50%) or above**, the label is **FAKE**. Below 50% → **REAL**.

---

### Confidence (the big % under REAL/FAKE)

**What it is:**  
The average “fake probability” across all sequences, shown as a percentage.

**How it is calculated:**

```
confidence = (sequence score 1 + sequence score 2 + … + sequence score N) ÷ N
```

**How to explain it to a lecturer:**  
“If confidence is 82%, the model averaged 82% ‘fake’ across all short clips it checked. It is not 82% sure in a legal sense — it is the model’s internal score after training on our dataset.”

---

### Blink rate (e.g. `0.3/s`)

**What it is:**  
How many **blinks per second** the system detected in the video.

**How it is calculated:**

1. For every sampled frame, compute **EAR** (eye openness).
2. If EAR drops **below 0.2**, treat the eye as **closed** (blink).
3. Count how many times the eye goes from **open → closed** (start of a blink).
4. Divide by video length in seconds (using ~25 frames per second as a time scale).

**Why we show it:**  
Real people blink naturally (often around 15–20 times per minute). Some deepfakes blink too rarely or in unnatural patterns. The model uses blink data internally.

**Example line for your presentation:**  
“A blink rate of 0.3 per second means about 18 blinks per minute, which is in a normal human range.”

---

### Sequences (e.g. `12`)

**What it is:**  
How many **8-frame clips** were extracted from your video and scored.

**How it is calculated:**  
After face detection and sampling, the video is cut into non-overlapping chunks of 8 eye frames. Each chunk = 1 sequence.

**Short video** → fewer sequences. **Long video** → more sequences.

---

### Avg score (average score)

**What it is:**  
The same idea as **confidence**, shown again in the stat cards — the **mean of all sequence scores** as a percentage.

**Formula:**

```
avg score = round( mean(all sequence scores) × 100 )
```

On the UI, **confidence** and **avg score** should match (or be very close). Both are the average fake probability.

---

### Sequence scores (the chart)

**What it is:**  
A line chart where **each point is one sequence** and the height is that sequence’s **fake probability** (0–100%).

**How to read it:**

- **Low points (below 50%)** — that part of the video looked more “real” to the model.
- **High points (above 50%)** — that part looked more “fake”.
- **Red dots** — scores at or above 50%.
- **Dashed line at 50%** — the threshold used for the final REAL/FAKE label.

**Why it is useful in a demo:**  
You can say: “Most of the video scored low, but sequence 7 spiked — that short section drove the average up.” This shows the system is **time-aware**, not a single-frame guess.

---

## A simple diagram you can draw on a whiteboard

```
Video upload
    ↓
Find face & eyes (MediaPipe)
    ↓
Sample ~10 frames/sec → crop eyes + measure EAR (blink signal)
    ↓
Group into sequences (8 frames each)
    ↓
For each sequence → Model → sequence score (% fake)
    ↓
Average all scores → Confidence
    ↓
≥ 50% average? → FAKE : REAL
    ↓
Also: count blinks from EAR → Blink rate
```

---

## What to say while the spinner says “Analysing…”

You can narrate live:

1. “The server is reading the video.”
2. “It is detecting the face and tracking the eyes on each frame.”
3. “It is measuring how open the eyes are — that’s how we get blink information.”
4. “It is cutting the video into short sequences of eight frames.”
5. “Each sequence goes through our trained LRCN-ViT model — CNN plus LSTM plus blink features.”
6. “We average the scores and show REAL or FAKE.”

Typical wait time on a laptop CPU: **30 seconds to a few minutes**, depending on video length.

---

## Honest limitations (good to mention to lecturers)

- The model was trained on a **limited dataset** , so accuracy is **not production-grade**.
- If **no face** is detected, you may get **UNKNOWN** or poor results.
- **Blink rate** is a simple counter; the main decision comes from the **neural network**, not a hand-written rule like “if blinks < X then fake.”
- **Lighting, angle, and sunglasses** can affect face detection and scores.
- This project focuses on **research and demonstration**, not forensic proof in court.

---

## One-minute script you can memorize

> “When I upload a video, the system samples frames and uses MediaPipe to find the eyes. For each frame it crops the eye region and measures the Eye Aspect Ratio to track blinks. The video is split into short sequences of eight frames. Our LRCN-ViT model — a Vision Transformer for spatial features plus an LSTM for time, plus blink features — scores each sequence from zero to one hundred percent fake. We average those scores for confidence and call it FAKE if the average is fifty percent or higher. Blink rate is shown separately: we count how often the eyes close per second. The chart lets us see which part of the video looked most suspicious. This implements our project idea: deepfake detection using eye-blink patterns and temporal learning, not just a single screenshot.”

---

