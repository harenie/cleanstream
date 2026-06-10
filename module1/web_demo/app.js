const form = document.querySelector("#moduleForm");
const statusBox = document.querySelector("#status");
const processButton = document.querySelector("#processButton");

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
    student_answer: document.querySelector("#studentAnswer").value,
    model_answer: document.querySelector("#modelAnswer").value,
    use_trained_model: document.querySelector("#useConceptModel").checked,
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

function renderResult(result) {
  const summary = result.summary;
  const conceptCoverage = numberValue(summary.concept_coverage_ratio);
  const semanticSimilarity = numberValue(summary.semantic_similarity_score);
  const languageQuality = numberValue(summary.language_quality_score);

  fields.conceptCoverage.textContent = formatScore(conceptCoverage);
  fields.semanticSimilarity.textContent = formatScore(semanticSimilarity);
  fields.languageQuality.textContent = formatScore(languageQuality);
  fields.reasoningQuality.textContent = titleCase(summary.reasoning_quality || "-");
  fields.reasoningDetail.textContent = `${summary.reasoning_connective_count ?? 0} markers`;
  fields.contradictionFlag.textContent = formatFlag(summary.contradiction_detected);
  fields.crossQuestionFlag.textContent = formatFlag(summary.cross_question_flag);
  fields.spellingCount.textContent = String(summary.spelling_error_count ?? 0);
  fields.wordCount.textContent = String(summary.answer_word_count ?? 0);

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

function titleCase(value) {
  return String(value).replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
}

function setStatus(text, state) {
  statusBox.textContent = text;
  statusBox.className = `status ${state || ""}`.trim();
}
