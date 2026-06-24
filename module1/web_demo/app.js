const form = document.querySelector("#moduleForm");
const statusBox = document.querySelector("#status");
const processButton = document.querySelector("#processButton");
const pathToggle = document.querySelector("#useLlmPath");
const pathModeLabel = document.querySelector("#pathModeLabel");

const fields = {
  conceptCoverage: document.querySelector("#conceptCoverage"),
  semanticSimilarity: document.querySelector("#semanticSimilarity"),
  languageQuality: document.querySelector("#languageQuality"),
  reasoningQuality: document.querySelector("#reasoningQuality"),
  reasoningDetail: document.querySelector("#reasoningDetail"),
  contradictionFlag: document.querySelector("#contradictionFlag"),
  crossQuestionFlag: document.querySelector("#crossQuestionFlag"),
  spellingCount: document.querySelector("#spellingCount"),
  wordCount: document.querySelector("#wordCount"),
  conceptBackend: document.querySelector("#conceptBackend"),
  reasoningBackend: document.querySelector("#reasoningBackend"),
  conceptCoverageBar: document.querySelector("#conceptCoverageBar"),
  semanticSimilarityBar: document.querySelector("#semanticSimilarityBar"),
  languageQualityBar: document.querySelector("#languageQualityBar"),
  coveredConcepts: document.querySelector("#coveredConcepts"),
  partialConcepts: document.querySelector("#partialConcepts"),
  missingConcepts: document.querySelector("#missingConcepts"),
  rawOutput: document.querySelector("#rawOutput"),
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Processing", "busy");
  processButton.disabled = true;

  const payload = {
    question: document.querySelector("#question").value,
    reasoning_requirement: document.querySelector("#reasoningRequirement").value,
    student_answer: document.querySelector("#studentAnswer").value,
    model_answer: document.querySelector("#modelAnswer").value,
    processing_path: pathToggle.checked ? "llm" : "fallback",
  };

  try {
    const response = await fetch("/api/module1-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Request failed.");
    }
    renderResult(result);
    setStatus("Complete", "done");
  } catch (error) {
    setStatus("Error", "error");
    fields.rawOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  } finally {
    processButton.disabled = false;
  }
});

pathToggle.addEventListener("change", updatePathModeLabel);
updatePathModeLabel();

function renderResult(result) {
  const summary = result.summary;
  const conceptCoverage = numberValue(summary.concept_coverage_ratio);
  const semanticSimilarity = numberValue(summary.semantic_similarity_score);
  const languageQuality = numberValue(summary.language_quality_score);

  fields.conceptCoverage.textContent = formatScore(conceptCoverage);
  fields.semanticSimilarity.textContent = formatScore(semanticSimilarity);
  fields.languageQuality.textContent = formatScore(languageQuality);
  fields.reasoningQuality.textContent = titleCase(summary.reasoning_quality || "-");
  fields.reasoningDetail.textContent = formatReasoningDetail(summary);
  fields.contradictionFlag.textContent = formatFlag(summary.contradiction_detected);
  fields.crossQuestionFlag.textContent = formatFlag(summary.cross_question_flag);
  fields.spellingCount.textContent = String(summary.spelling_error_count ?? 0);
  fields.wordCount.textContent = String(summary.answer_word_count ?? 0);
  fields.conceptBackend.textContent = formatBackend(summary.concept_backend);
  fields.reasoningBackend.textContent = formatBackend(summary.reasoning_backend);

  setMeter(fields.conceptCoverageBar, conceptCoverage);
  setMeter(fields.semanticSimilarityBar, semanticSimilarity);
  setMeter(fields.languageQualityBar, languageQuality);

  renderList(fields.coveredConcepts, result.concepts.present);
  renderList(fields.partialConcepts, result.concepts.partial);
  renderList(fields.missingConcepts, result.concepts.missing);
  fields.rawOutput.textContent = JSON.stringify(result.raw, null, 2);
}

function renderList(target, items) {
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "None";
    target.appendChild(empty);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  }
}

function setMeter(target, value) {
  const bounded = Math.max(0, Math.min(1, numberValue(value)));
  target.style.width = `${Math.round(bounded * 100)}%`;
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function formatScore(value) {
  return numberValue(value).toFixed(3);
}

function formatFlag(value) {
  return value ? "Yes" : "No";
}

function formatBackend(value) {
  return titleCase(String(value || "-").replace(/[-_]/g, " "));
}

function formatReasoningDetail(summary) {
  const markerText = `${summary.reasoning_connective_count ?? 0} markers`;
  if (summary.reasoning_required === false) {
    return summary.reasoning_skip_reason || `Not required · ${markerText}`;
  }
  const expectedType = summary.reasoning_expected_type
    ? `${titleCase(String(summary.reasoning_expected_type).replace(/_/g, " "))} · `
    : "";
  if (!summary.reasoning_model_label) {
    return `${expectedType}${markerText}`;
  }
  const confidence = Number(summary.reasoning_model_confidence);
  const confidenceText = Number.isFinite(confidence) ? ` ${confidence.toFixed(3)}` : "";
  return `${expectedType}${summary.reasoning_model_label}${confidenceText} · ${markerText}`;
}

function titleCase(value) {
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
}

function setStatus(text, state) {
  statusBox.textContent = text;
  statusBox.className = `status ${state || ""}`.trim();
}

function updatePathModeLabel() {
  pathModeLabel.textContent = pathToggle.checked ? "LLM path" : "Fallback path";
}
