# Proofline web demo

Proofline is a standalone frontend for the handwritten-math-to-LaTeX model. It
currently runs in an explicitly labelled demo mode while the fine-tuned GLM-OCR
checkpoint is pending.

## Run locally

From the repository root:

```bash
python3 -m http.server 8080 --directory site
```

Open <http://localhost:8080>.

The demo supports:

- PDF, PNG, JPG, and WebP upload with local source preview;
- deterministic placeholder transcription;
- editable LaTeX with live MathJax rendering;
- copy and `.tex` download actions;
- responsive Source / LaTeX / Preview tabs on smaller screens.

No uploaded file leaves the browser while `demoMode` is `true`.

## Connect the GLM backend

Edit `config.js`:

```js
window.PROOFLINE_CONFIG = Object.freeze({
  apiBaseUrl: "https://your-model-service.example",
  demoMode: false,
  modelName: "Fine-tuned GLM-OCR",
  modelVersion: "your-frozen-checkpoint-hash",
  maxFileBytes: 20 * 1024 * 1024,
  pollIntervalMs: 1200,
  maxPollAttempts: 150,
});
```

The frontend expects this asynchronous API:

### Create a transcription

`POST /api/v1/transcriptions` as multipart form data:

- `file`: required PDF or image;
- `mode`: `document` or `formula`.

Return HTTP `202`:

```json
{
  "id": "tr_example",
  "status": "queued",
  "model": {
    "id": "glm-math-latex",
    "version": "checkpoint-hash"
  },
  "progress": {
    "completed_pages": 0,
    "total_pages": null
  },
  "pages": []
}
```

### Poll the transcription

`GET /api/v1/transcriptions/{id}` returns the same object with status
`queued`, `processing`, `completed`, or `failed`. A completed response contains:

```json
{
  "id": "tr_example",
  "status": "completed",
  "model": {
    "id": "glm-math-latex",
    "version": "checkpoint-hash"
  },
  "progress": {
    "completed_pages": 1,
    "total_pages": 1
  },
  "pages": [
    {
      "number": 1,
      "latex": "\\begin{aligned}...\\end{aligned}",
      "warnings": [],
      "hit_token_cap": false
    }
  ],
  "document_latex": "\\begin{aligned}...\\end{aligned}",
  "demo": false,
  "error": null
}
```

Keep API credentials and model paths on the server. The production service
should verify file magic bytes, reject encrypted PDFs, constrain page/pixel
counts, isolate rendering, rate-limit requests, and automatically delete
uploads and results. Never compile model-produced LaTeX with unrestricted
shell access.

## Benchmark placeholders

The comparison section intentionally contains no invented values. Populate it
only after the fine-tuned GLM checkpoint and frontier reference have been run
against the same leakage-filtered set and evaluation rubric.
