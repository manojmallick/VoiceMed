# VoiceMed — Demo Video Production Guide
## 90-Second Keynote-Ready Demo · Gemma 4 Good Hackathon

---

## Equipment needed (all free)

- Your laptop or phone (the one running VoiceMed)
- A second phone as the camera (or screen recording software)
- OBS Studio (free) or QuickTime for screen recording
- A printed prop referral letter (print the PDF output)
- Optional: a second phone showing "No internet connection" clearly

---

## Pre-filming checklist

- [ ] VoiceMed app running locally at localhost:7860
- [ ] Gemma 4 E4B model loaded and tested (first inference is slow — warm it up)
- [ ] Airplane mode tested — confirmed app works offline
- [ ] A real wound photo ready (use the test image from data/test_cases/ or a stock medical image)
- [ ] Audio description pre-planned (practice it 3 times before recording)
- [ ] PDF referral letter pre-generated and printed
- [ ] Clean browser window — hide bookmarks bar, use incognito for a clean UI
- [ ] Screen resolution set to 1080p minimum
- [ ] Notification banners turned OFF on your device

---

## Shot list — 90 seconds total

### SHOT 1 — The problem hook (0:00–0:10)

**What to show:** Simple text slide or title card
**Text:** "800 million people live more than 1 hour from a hospital"
**How to film:** Create in Google Slides, record screen, or use a simple HTML page with the text centered on black background

**Script (voiceover or text overlay):**
> "800 million people live more than 1 hour from a hospital. Their only point of care is a community health worker — with a phone."

**Technical:** 10 seconds, static card, white text on dark background

---

### SHOT 2 — The offline moment (0:10–0:22) ← MOST IMPORTANT SHOT

**What to show:**
1. Pull up phone Settings → Wi-Fi → toggle OFF (camera watches this)
2. Pull up mobile data → toggle OFF
3. Show status bar clearly: airplane icon or "No SIM / No internet"
4. Then show VoiceMed open in browser — it loads from localhost, no internet needed

**How to film:** Use a second phone as the camera pointed at your primary phone/laptop screen

**Script:**
> "VoiceMed runs with zero internet. No cloud. No subscription. Just Gemma 4 on the device."

**Why this shot wins:** This 12-second moment proves the entire offline claim visually. No amount of text or slides can replace this. Judges who see airplane mode active + app running will immediately understand the impact.

**Technical tip:** Make sure your screen brightness is high. Film in a dim room so the screen is clearly visible. Consider doing this twice and using the cleaner take.

---

### SHOT 3 — Photo the patient (0:22–0:32)

**What to show:**
- The Gradio UI open on screen
- Upload a wound photograph using the image upload button
- The image appears in the interface

**What image to use:** Use a clearly recognizable stock medical image (skin wound or rash). Kaggle has medical imaging datasets. Alternatively use any photograph of a minor cut — your own hand works fine.

**Script:**
> "The health worker photographs the patient's condition."

**Technical:** Keep the UI visible. Make sure the image thumbnail is clearly visible after upload.

---

### SHOT 4 — Voice input (0:32–0:45)

**What to show:**
- Click the microphone button in Gradio
- Speak clearly into the mic (or upload a pre-recorded audio file)
- The waveform/loading indicator shows it's processing

**What to say (your audio description — speak slowly):**
> "Patient is a 34-year-old male farmer. Deep laceration on the right forearm from a machete. Wound is approximately five centimetres. Bleeding has slowed but not fully stopped. Patient is alert and speaking. No prior tetanus vaccination."

**Script overlay:**
> "Then describes symptoms in any language — Gemma 4 understands 140+ languages."

**Technical:** If Gradio's microphone doesn't record cleanly, pre-record the audio as a WAV file and upload it. The upload path produces identical results and is more reliable for filming.

---

### SHOT 5 — The assessment appearing (0:45–1:00)

**What to show:**
- Click the "Assess Patient" button
- Show a loading indicator (Gradio shows "Running..." — this is good, shows it's real inference)
- The triage result appears on screen:
  - Severity: REFER_ROUTINE (orange/amber)
  - Primary concern: "Deep laceration requiring wound closure and tetanus prophylaxis"
  - Recommended actions: Clean wound, apply antiseptic, close wound, give paracetamol, arrange clinic visit for suturing
  - Patient advice: "Your wound needs to be closed by a nurse or doctor. Go to the nearest clinic today."

**Script:**
> "In 90 seconds: severity level, recommended actions, and patient advice — all offline, all from Gemma 4."

**Technical:** If inference takes longer than 30 seconds on camera, use a faster model (E2B) for the demo video, and note in your submission that E4B is the deployment target. Speed vs quality trade-off is normal and judges understand this.

---

### SHOT 6 — The referral letter (1:00–1:10)

**What to show:**
- Expand the "Referral Letter" accordion in the Gradio UI
- Show the full letter text on screen
- OR show the printed PDF prop in your hand

**Script:**
> "A referral letter is generated automatically — the patient carries it to the clinic."

**Technical:** Pre-generate a referral letter and print it on paper. Hold it up to the camera for 3 seconds — a physical printout is more visceral than text on screen.

---

### SHOT 7 — The impact close (1:10–1:25)

**What to show:**
- Kaggle notebook open — show "Run All" executing
- GitHub repo page — show the file structure briefly
- Final text card: "Apache 2.0 · Gemma 4 E4B · Works offline · 1.2M CHWs worldwide"

**Script:**
> "$0 per query. No internet required. One-click install for any NGO. Full code, open source, Apache 2.0 — built for the Gemma 4 Good Hackathon."

**Technical:** Show the GitHub repo briefly to signal reproducibility. 3 seconds is enough.

---

## Multilingual bonus shot (add this if time permits — worth extra points)

**After Shot 4, add this 8-second insert:**
- Type a Swahili description into the text box:
  > "Mtoto wa miaka 3, ana homa kali, hakunywa maji"
- Show the result come back in Swahili or English (either is fine)
- Text overlay: "140+ languages — no translation needed"

---

## Upload and submit

1. Upload to YouTube as **Unlisted** (not private — judges need to see it without logging in)
2. Title: "VoiceMed — Offline Community Health Triage with Gemma 4 | Gemma 4 Good Hackathon"
3. Description: Include your GitHub link and Kaggle notebook link
4. Copy the YouTube URL — this goes in your Kaggle hackathon submission form

---

## Common filming mistakes to avoid

| Mistake | Fix |
|---|---|
| Showing a blank/loading screen for too long | Warm up the model before filming — first inference is always slow |
| Airplane mode not visible in frame | Use a second phone as camera, zoom in on status bar |
| Audio is muffled | Speak directly into mic, record in a quiet room |
| Demo shows "Error" or crash | Test the full flow 3 times before filming |
| Video is too long (>2 minutes) | Edit ruthlessly — judges watch hundreds of videos |
| No voiceover or text overlays | Add text overlays in iMovie/CapCut/DaVinci Resolve (all free) |
| GitHub repo shown but it's private | Make repo public before filming this shot |

---

## Editing tools (all free)

- **CapCut** (mobile) — fastest for adding text overlays
- **DaVinci Resolve** (desktop) — professional, free
- **iMovie** (Mac) — simple, fast
- **OBS Studio** — screen recording with scene transitions

---

## Final video specs for YouTube upload

- Resolution: 1080p (1920×1080)
- Duration: 90 seconds (absolute max: 2 minutes)
- Format: MP4 (H.264)
- Title format: "VoiceMed — [your tagline] | Gemma 4 Good Hackathon 2026"
