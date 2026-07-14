(() => {
  "use strict";

  const config = window.PROOFLINE_CONFIG || {
    apiBaseUrl: "",
    demoMode: true,
    modelName: "Fine-tuned GLM-OCR",
    modelVersion: "checkpoint-pending",
    maxFileBytes: 20 * 1024 * 1024,
    pollIntervalMs: 1200,
    maxPollAttempts: 150,
  };

  const demoLatex = {
    document: String.raw`\begin{aligned}
f(x) &= x^2 + 3x \\
f'(x) &= 2x + 2 \\
0 &= 2x + 2 \\
x &= -1
\end{aligned}`,
    formula: String.raw`\int_0^1 \left(3x^2 + 2x\right)\,dx
= \left[x^3 + x^2\right]_0^1
= 2`,
  };

  const allowedExtensions = new Set(["pdf", "png", "jpg", "jpeg", "webp"]);
  const allowedMimeTypes = new Set([
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
  ]);

  const elements = {
    dropZone: document.querySelector("#dropZone"),
    fileInput: document.querySelector("#fileInput"),
    transcribeButton: document.querySelector("#transcribeButton"),
    sampleButton: document.querySelector("#sampleButton"),
    fileBar: document.querySelector("#fileBar"),
    fileType: document.querySelector("#fileType"),
    fileName: document.querySelector("#fileName"),
    fileMeta: document.querySelector("#fileMeta"),
    fileStatus: document.querySelector("#fileStatus"),
    errorBanner: document.querySelector("#errorBanner"),
    sourceStage: document.querySelector("#sourceStage"),
    sourceImage: document.querySelector("#sourceImage"),
    sourcePdf: document.querySelector("#sourcePdf"),
    activeScanLine: document.querySelector("#activeScanLine"),
    latexEditor: document.querySelector("#latexEditor"),
    characterCount: document.querySelector("#characterCount"),
    mathOutput: document.querySelector("#mathOutput"),
    previewState: document.querySelector("#previewState"),
    runtimeBadge: document.querySelector("#runtimeBadge"),
    resultLabel: document.querySelector("#resultLabel"),
    resultDescription: document.querySelector("#resultDescription"),
    copyButton: document.querySelector("#copyButton"),
    downloadButton: document.querySelector("#downloadButton"),
    resetButton: document.querySelector("#resetButton"),
    zoomOutButton: document.querySelector("#zoomOutButton"),
    zoomInButton: document.querySelector("#zoomInButton"),
    zoomLabel: document.querySelector("#zoomLabel"),
    rotateButton: document.querySelector("#rotateButton"),
    mobileTabs: [...document.querySelectorAll(".mobile-tabs [role='tab']")],
    panes: [...document.querySelectorAll(".workbench-pane")],
    processSteps: [...document.querySelectorAll(".process-rail li")],
  };

  const state = {
    file: null,
    objectUrl: null,
    isSample: true,
    rotation: 0,
    zoom: 1,
    renderingTimer: null,
    isTranscribing: false,
  };

  function init() {
    bindUploadEvents();
    bindWorkspaceEvents();
    configureRuntimeLabel();
    updateCharacterCount();
    updateSourceTransform();
    const initialRender = renderLatex();
    restoreHashPositionAfterFonts(initialRender);
  }

  function restoreHashPositionAfterFonts(renderReady = Promise.resolve()) {
    if (!window.location.hash) return;
    const alignTarget = () => {
      const target = document.querySelector(window.location.hash);
      target?.scrollIntoView({ block: "start" });
    };
    const fontsReady = document.fonts?.ready || Promise.resolve();
    Promise.all([fontsReady, renderReady]).then(() => window.setTimeout(alignTarget, 50));
  }

  function bindUploadEvents() {
    elements.dropZone.addEventListener("click", () => elements.fileInput.click());
    elements.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });

    elements.fileInput.addEventListener("change", (event) => {
      const [file] = event.target.files;
      if (file) loadFile(file);
      event.target.value = "";
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.dropZone.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.dropZone.classList.remove("is-dragging");
      });
    });

    elements.dropZone.addEventListener("drop", (event) => {
      const [file] = event.dataTransfer.files;
      if (file) loadFile(file);
    });
  }

  function bindWorkspaceEvents() {
    elements.transcribeButton.addEventListener("click", transcribeCurrentFile);
    elements.sampleButton.addEventListener("click", loadSample);
    elements.resetButton.addEventListener("click", loadSample);
    elements.copyButton.addEventListener("click", copyLatex);
    elements.downloadButton.addEventListener("click", downloadLatex);

    elements.latexEditor.addEventListener("input", () => {
      updateCharacterCount();
      window.clearTimeout(state.renderingTimer);
      state.renderingTimer = window.setTimeout(renderLatex, 220);
    });

    elements.zoomOutButton.addEventListener("click", () => {
      state.zoom = Math.max(0.6, state.zoom - 0.1);
      updateSourceTransform();
    });

    elements.zoomInButton.addEventListener("click", () => {
      state.zoom = Math.min(2, state.zoom + 0.1);
      updateSourceTransform();
    });

    elements.rotateButton.addEventListener("click", () => {
      state.rotation = (state.rotation + 90) % 360;
      updateSourceTransform();
    });

    elements.mobileTabs.forEach((tab) => {
      tab.addEventListener("click", () => activateMobilePane(tab.dataset.paneTarget));
    });
  }

  function configureRuntimeLabel() {
    const title = elements.runtimeBadge.querySelector("span");
    const value = elements.runtimeBadge.querySelector("strong");
    if (!config.demoMode && config.apiBaseUrl) {
      title.textContent = config.modelName;
      value.textContent = `Connected · ${config.modelVersion}`;
      return;
    }
    title.textContent = "Demo runtime";
    value.textContent = "Placeholder output";
  }

  function getExtension(filename) {
    const pieces = filename.toLowerCase().split(".");
    return pieces.length > 1 ? pieces.pop() : "";
  }

  function validateFile(file) {
    const extension = getExtension(file.name);
    const allowedType = allowedMimeTypes.has(file.type) || allowedExtensions.has(extension);

    if (!allowedType) {
      return "Choose a PDF, PNG, JPG, or WebP file.";
    }
    if (file.size > config.maxFileBytes) {
      const limit = Math.round(config.maxFileBytes / (1024 * 1024));
      return `This file is too large. Choose a file smaller than ${limit} MB.`;
    }
    if (file.size === 0) {
      return "This file is empty. Choose a file that contains a scan or photo.";
    }
    return null;
  }

  function loadFile(file) {
    const validationError = validateFile(file);
    if (validationError) {
      showError(validationError);
      return;
    }

    clearError();
    releaseObjectUrl();
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);
    state.isSample = false;
    state.rotation = 0;
    state.zoom = 1;
    updateSourceTransform();

    const extension = getExtension(file.name);
    const isPdf = file.type === "application/pdf" || extension === "pdf";
    if (isPdf) {
      elements.sourceImage.hidden = true;
      elements.sourcePdf.hidden = false;
      elements.sourcePdf.data = state.objectUrl;
      elements.sourcePdf.setAttribute("aria-label", `PDF preview of ${file.name}`);
    } else {
      elements.sourcePdf.hidden = true;
      elements.sourcePdf.removeAttribute("data");
      elements.sourceImage.hidden = false;
      elements.sourceImage.src = state.objectUrl;
      elements.sourceImage.alt = `Preview of ${file.name}`;
    }

    elements.fileType.textContent = (extension || "FILE").slice(0, 5).toUpperCase();
    elements.fileName.textContent = file.name;
    elements.fileMeta.textContent = `${formatBytes(file.size)} · stored in this browser during demo mode`;
    setStatus("Ready to transcribe", "ready");
    setProcessStep("upload");
    elements.resultLabel.textContent = "Previous illustrative output.";
    elements.resultDescription.textContent = "Select Transcribe handwriting to replace it with a demo response.";
    activateMobilePane("sourcePane");
  }

  function loadSample() {
    releaseObjectUrl();
    clearError();
    state.file = null;
    state.isSample = true;
    state.rotation = 0;
    state.zoom = 1;
    elements.sourcePdf.hidden = true;
    elements.sourcePdf.removeAttribute("data");
    elements.sourceImage.hidden = false;
    elements.sourceImage.src = "sample-handwriting.svg";
    elements.sourceImage.alt = "Sample handwritten derivative with a visible plus two term";
    elements.fileType.textContent = "SVG";
    elements.fileName.textContent = "sample-derivative.svg";
    elements.fileMeta.textContent = "Illustrative sample · not uploaded";
    elements.latexEditor.value = demoLatex.document;
    elements.resultLabel.textContent = "Illustrative output.";
    elements.resultDescription.textContent = "The connected GLM model will replace this sample response.";
    setStatus("Ready to transcribe", "ready");
    setProcessStep("upload");
    updateSourceTransform();
    updateCharacterCount();
    renderLatex();
    activateMobilePane("sourcePane");
  }

  async function transcribeCurrentFile() {
    if (state.isTranscribing) return;
    clearError();
    state.isTranscribing = true;
    elements.transcribeButton.disabled = true;
    elements.transcribeButton.querySelector("span").textContent = "Reading source…";
    elements.activeScanLine.classList.add("is-scanning");
    setStatus("Reading the source", "processing");
    setProcessStep("transcribe");

    try {
      let result;
      if (config.demoMode || !config.apiBaseUrl) {
        result = await runDemoTranscription();
      } else {
        if (!state.file) {
          throw new Error("Upload a PDF or image before using the connected model.");
        }
        result = await runApiTranscription(state.file, getSelectedMode());
      }

      elements.latexEditor.value = result.latex;
      updateCharacterCount();
      await renderLatex();
      setStatus("Transcription ready for review", "done");
      setProcessStep("review");
      elements.resultLabel.textContent = result.demo ? "Illustrative demo output." : `${result.modelName} output.`;
      elements.resultDescription.textContent = result.demo
        ? "No model processed this file. Connect the GLM endpoint to replace this placeholder."
        : "Review every line against the source before exporting.";
      activateMobilePane("editorPane");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Transcription failed. Try the file again.";
      showError(message);
      setStatus("Transcription needs attention", "error");
      setProcessStep("upload");
    } finally {
      state.isTranscribing = false;
      elements.transcribeButton.disabled = false;
      elements.transcribeButton.querySelector("span").textContent = "Transcribe handwriting";
      elements.activeScanLine.classList.remove("is-scanning");
    }
  }

  async function runDemoTranscription() {
    setStatus("Mapping visible symbols", "processing");
    await sleep(620);
    elements.transcribeButton.querySelector("span").textContent = "Preserving notation…";
    setStatus("Preserving the writer’s notation", "processing");
    await sleep(760);
    elements.transcribeButton.querySelector("span").textContent = "Rendering proof…";
    setStatus("Preparing editable LaTeX", "processing");
    await sleep(620);
    return {
      latex: demoLatex[getSelectedMode()],
      demo: true,
      modelName: config.modelName,
    };
  }

  async function runApiTranscription(file, mode) {
    const base = config.apiBaseUrl.replace(/\/$/, "");
    const body = new FormData();
    body.append("file", file);
    body.append("mode", mode);

    const response = await fetch(`${base}/api/v1/transcriptions`, {
      method: "POST",
      body,
    });
    const payload = await parseJsonResponse(response);

    if (!response.ok) {
      throw new Error(payload?.error?.message || `The model service returned ${response.status}.`);
    }

    let completed = payload;
    if (payload.status !== "completed") {
      completed = await pollTranscription(base, payload.id);
    }
    if (!completed.document_latex && !completed.pages?.length) {
      throw new Error("The model completed without returning LaTeX.");
    }

    const latex = completed.document_latex || completed.pages.map((page) => page.latex).join("\n\n\\newpage\n\n");
    return {
      latex,
      demo: Boolean(completed.demo),
      modelName: completed.model?.id || config.modelName,
    };
  }

  async function pollTranscription(base, id) {
    if (!id) throw new Error("The model service did not return a transcription ID.");

    for (let attempt = 0; attempt < config.maxPollAttempts; attempt += 1) {
      await sleep(config.pollIntervalMs);
      const response = await fetch(`${base}/api/v1/transcriptions/${encodeURIComponent(id)}`);
      const payload = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(payload?.error?.message || `Unable to read transcription status (${response.status}).`);
      }
      if (payload.status === "completed") return payload;
      if (payload.status === "failed") {
        throw new Error(payload?.error?.message || "The model could not transcribe this file.");
      }

      const progress = payload.progress;
      if (progress?.total_pages) {
        setStatus(`Reading page ${progress.completed_pages + 1} of ${progress.total_pages}`, "processing");
      } else {
        setStatus("The model is processing the upload", "processing");
      }
    }
    throw new Error("The transcription timed out. Try again with a smaller file.");
  }

  async function parseJsonResponse(response) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  function getSelectedMode() {
    return document.querySelector("input[name='mode']:checked")?.value || "document";
  }

  async function renderLatex() {
    const latex = elements.latexEditor.value.trim();
    elements.previewState.textContent = "Rendering";

    if (!latex) {
      elements.mathOutput.textContent = "The rendered proof will appear here as you add LaTeX.";
      elements.mathOutput.classList.add("is-empty");
      elements.previewState.textContent = "Empty";
      return;
    }

    elements.mathOutput.classList.remove("is-empty");
    const mathSource = `\\[${latex}\\]`;
    elements.mathOutput.textContent = mathSource;

    try {
      if (!window.MathJax?.typesetPromise) {
        elements.previewState.textContent = "Source view";
        return;
      }
      window.MathJax.typesetClear?.([elements.mathOutput]);
      await window.MathJax.typesetPromise([elements.mathOutput]);
      elements.previewState.textContent = "Rendered";
    } catch {
      elements.mathOutput.textContent = latex;
      elements.previewState.textContent = "Check syntax";
    }
  }

  async function copyLatex() {
    const text = elements.latexEditor.value;
    if (!text) return;
    const original = elements.copyButton.textContent;

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      elements.latexEditor.focus();
      elements.latexEditor.select();
      document.execCommand("copy");
      elements.latexEditor.setSelectionRange(0, 0);
    }

    elements.copyButton.textContent = "Copied";
    window.setTimeout(() => {
      elements.copyButton.textContent = original;
    }, 1400);
  }

  function downloadLatex() {
    const latex = elements.latexEditor.value;
    if (!latex) return;
    const sourceName = state.file?.name || "proofline-transcription";
    const safeBase = sourceName.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "");
    const blob = new Blob([latex, "\n"], { type: "application/x-tex;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeBase || "proofline-transcription"}.tex`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function activateMobilePane(targetId) {
    elements.mobileTabs.forEach((tab) => {
      const isTarget = tab.dataset.paneTarget === targetId;
      tab.classList.toggle("is-active", isTarget);
      tab.setAttribute("aria-selected", String(isTarget));
    });
    elements.panes.forEach((pane) => pane.classList.toggle("is-active", pane.id === targetId));
  }

  function updateSourceTransform() {
    elements.sourceStage.style.setProperty("--source-rotation", `${state.rotation}deg`);
    elements.sourceStage.style.setProperty("--source-scale", state.zoom.toFixed(1));
    elements.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function updateCharacterCount() {
    const length = elements.latexEditor.value.length;
    elements.characterCount.textContent = `${length.toLocaleString()} character${length === 1 ? "" : "s"}`;
  }

  function setStatus(message, variant) {
    elements.fileStatus.lastChild.textContent = ` ${message}`;
    elements.fileStatus.classList.remove("is-processing", "is-done", "is-error");
    if (variant === "processing") elements.fileStatus.classList.add("is-processing");
    if (variant === "done") elements.fileStatus.classList.add("is-done");
    if (variant === "error") elements.fileStatus.classList.add("is-error");
  }

  function setProcessStep(currentStep) {
    const order = ["upload", "transcribe", "review"];
    const currentIndex = order.indexOf(currentStep);
    elements.processSteps.forEach((step, index) => {
      step.classList.toggle("is-current", index === currentIndex);
      step.classList.toggle("is-complete", index < currentIndex);
    });
  }

  function showError(message) {
    elements.errorBanner.textContent = message;
    elements.errorBanner.hidden = false;
  }

  function clearError() {
    elements.errorBanner.hidden = true;
    elements.errorBanner.textContent = "";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function releaseObjectUrl() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }

  function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  window.addEventListener("beforeunload", releaseObjectUrl);
  init();
})();
