/*
 * Proofline runtime configuration.
 *
 * Leave apiBaseUrl blank while the fine-tuned GLM checkpoint is pending.
 * When the backend is ready, point this at the service implementing the API
 * contract documented in site/README.md and set demoMode to false.
 */
window.PROOFLINE_CONFIG = Object.freeze({
  apiBaseUrl: "",
  demoMode: true,
  modelName: "Fine-tuned GLM-OCR",
  modelVersion: "checkpoint-pending",
  maxFileBytes: 20 * 1024 * 1024,
  pollIntervalMs: 1200,
  maxPollAttempts: 150,
});
