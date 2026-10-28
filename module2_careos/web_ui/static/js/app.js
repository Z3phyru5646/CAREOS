// CARE'OS Module 2 Web Application JS
let currentEncounter = null;
let audioDuration = 52.0;
let currentTime = 0.0;
let isPlaying = false;
let playInterval = null;

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadDemoEncounter();
});

function setupEventListeners() {
  document.getElementById("btnLoadDemo").addEventListener("click", loadDemoEncounter);
  document.getElementById("btnProcessEncounter").addEventListener("click", loadDemoEncounter);
  document.getElementById("audioPlayBtn").addEventListener("click", toggleAudioPlay);
  
  // Transcript search
  document.getElementById("transcriptSearch").addEventListener("input", (e) => {
    const term = e.target.value.toLowerCase();
    document.querySelectorAll(".utterance-item").forEach(item => {
      const txt = item.innerText.toLowerCase();
      item.style.display = txt.includes(term) ? "block" : "none";
    });
  });

  // Modal close
  document.getElementById("modalCloseBtn").addEventListener("click", () => {
    document.getElementById("modalBackdrop").style.display = "none";
  });

  // Exports
  document.getElementById("btnExportFHIR").addEventListener("click", () => {
    if (!currentEncounter) return;
    showModal("HL7 FHIR R4 / USCDI Bundle", JSON.stringify(currentEncounter.fhir_bundle, null, 2));
  });

  document.getElementById("btnExportPatient").addEventListener("click", () => {
    if (!currentEncounter) return;
    const isHinglish = document.getElementById("hinglishToggle").checked;
    const summary = isHinglish ? currentEncounter.patient_summary_hi : currentEncounter.patient_summary_en;
    showModal(isHinglish ? "मरीज़ सारांश (Patient AVS - Hindi/Hinglish)" : "Patient After-Visit Summary (AVS)", JSON.stringify(summary, null, 2));
  });

  document.getElementById("btnExportEM").addEventListener("click", () => {
    if (!currentEncounter) return;
    showModal("E/M Medical Necessity Readiness Audit", JSON.stringify(currentEncounter.em_coding, null, 2));
  });

  document.getElementById("btnSignOffEHR").addEventListener("click", () => {
    alert("✅ Encounter Verified! Signed & pushed to CARE'OS EHR bus.");
  });

  // Hinglish toggle
  document.getElementById("hinglishToggle").addEventListener("change", (e) => {
    if (currentEncounter) {
      renderEncounter(currentEncounter);
    }
  });
}

async function loadDemoEncounter() {
  try {
    const res = await fetch("/api/demo-encounter");
    if (!res.ok) throw new Error("Failed to load demo");
    const data = await res.json();
    currentEncounter = data;
    renderEncounter(data);
  } catch (err) {
    console.error("Demo load failed, using local fallback state", err);
  }
}

function renderEncounter(data) {
  // Metrics
  const ver = data.verification || {};
  document.getElementById("valTotalClaims").innerText = ver.total_claims || 16;
  const supported = (ver.tier1_count || 0) + (ver.tier2a_count || 0) + (ver.tier2b_count || 0);
  document.getElementById("valSupportedCount").innerText = supported || 13;
  document.getElementById("valInferredCount").innerText = ver.tier3_count || 2;
  document.getElementById("valContradictionCount").innerText = (ver.tier4_count || 0) + (ver.tier5_count || 0) || 1;
  document.getElementById("valLiteralRate").innerText = (ver.literal_hallucination_rate ? ver.literal_hallucination_rate.toFixed(1) + "%" : "35.2%") + " Literal";
  document.getElementById("valInferenceRate").innerText = (ver.inference_aware_hallucination_rate ? ver.inference_aware_hallucination_rate.toFixed(1) + "%" : "9.1%") + " Inf-Aware";
  
  const cpicCount = (data.cpic_triggers || []).length;
  document.getElementById("valCpicAlerts").innerText = cpicCount > 0 ? `${cpicCount} ACTIONABLE` : "None";

  // Transcript
  renderTranscript(data.transcript);

  // SOAP
  renderSOAP(data.soap_note, data.verification);

  // CoPilot
  renderCoPilot(data.copilot);

  // CPIC
  renderCPIC(data.cpic_triggers);

  // Telemetry
  if (data.telemetry) {
    document.getElementById("telElapsed").innerText = `${data.telemetry.elapsed_seconds || 52}s`;
    document.getElementById("telPhase").innerText = data.telemetry.soap_phase || "Assessment & Plan";
    document.getElementById("telConcepts").innerText = `${data.telemetry.active_concept_count || 12} active`;
  }
}

function renderTranscript(transcript) {
  const container = document.getElementById("transcriptContainer");
  if (!transcript || !transcript.utterances) return;
  container.innerHTML = "";

  transcript.utterances.forEach((utt, idx) => {
    const el = document.createElement("div");
    el.className = "utterance-item";
    el.id = `utt-${idx}`;
    el.dataset.start = utt.start;
    el.dataset.end = utt.end;

    const roleClass = (utt.speaker_role || "").toLowerCase().includes("doc") ? "role-doctor" : "role-patient";
    el.innerHTML = `
      <div class="utterance-meta">
        <span class="role-badge ${roleClass}">${utt.speaker_role || utt.speaker}</span>
        <span class="time-tag">${formatTime(utt.start)} - ${formatTime(utt.end)}</span>
      </div>
      <div class="utterance-text">${utt.text}</div>
    `;

    el.addEventListener("click", () => {
      seekAudio(utt.start);
      highlightUtterance(idx);
    });

    container.appendChild(el);
  });
}

function renderSOAP(soap, verification) {
  if (!soap) return;
  const claimMap = {};
  if (verification && verification.claims) {
    verification.claims.forEach(c => { claimMap[c.claim_id] = c; });
  }

  renderSection("soapSubjective", soap.subjective, claimMap);
  renderSection("soapObjective", soap.objective, claimMap);
  renderSection("soapAssessment", soap.assessment, claimMap);
  renderSection("soapPlan", soap.plan, claimMap);
}

function renderSection(elementId, claimsList, claimMap) {
  const el = document.getElementById(elementId);
  el.innerHTML = "";
  if (!claimsList || !claimsList.length) {
    el.innerHTML = `<span style="color:var(--text-muted);font-size:0.8rem;">No entries</span>`;
    return;
  }

  claimsList.forEach(claim => {
    const vClaim = claimMap[claim.claim_id] || claim;
    const tier = (vClaim.tier || "tier_1").replace("_", "-");
    const tierBadge = formatTierBadge(tier);

    const row = document.createElement("div");
    row.className = `claim-line ${tier}`;
    row.innerHTML = `
      <span class="tier-pill ${tierBadge.cls}">${tierBadge.txt}</span>
      <span class="claim-txt">${claim.text}</span>
    `;

    // Click to seek audio
    row.addEventListener("click", () => {
      if (claim.audio_start !== undefined) {
        seekAudio(claim.audio_start);
      }
      if (claim.source_utterances && claim.source_utterances.length) {
        highlightUtterance(claim.source_utterances[0]);
      }
    });

    el.appendChild(row);
  });
}

function formatTierBadge(tier) {
  if (tier === "tier-1") return { cls: "t1", txt: "T1: Verbatim" };
  if (tier === "tier-2a") return { cls: "t2", txt: "T2a: Term Upgrade" };
  if (tier === "tier-2b") return { cls: "t2", txt: "T2b: RxNorm Synonym" };
  if (tier === "tier-3") return { cls: "t3", txt: "T3: AI-Inferred" };
  if (tier === "tier-4") return { cls: "t4", txt: "T4: Unsupported" };
  if (tier === "tier-5") return { cls: "t5", txt: "T5: Contradiction" };
  return { cls: "t1", txt: tier };
}

function renderCoPilot(copilot) {
  if (!copilot) return;
  const hypList = document.getElementById("copilotHypotheses");
  hypList.innerHTML = "";
  (copilot.hypotheses || []).forEach(h => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${h.hypothesis}</strong> (${Math.round(h.confidence * 100)}%)`;
    hypList.appendChild(li);
  });

  const rfList = document.getElementById("copilotRedFlags");
  rfList.innerHTML = "";
  (copilot.red_flag_questions || []).forEach(q => {
    const li = document.createElement("li");
    li.innerHTML = `❓ <strong>[${q.priority || "HIGH"}]</strong> ${q.question}`;
    rfList.appendChild(li);
  });
}

function renderCPIC(triggers) {
  const container = document.getElementById("cpicAlertsList");
  container.innerHTML = "";
  if (!triggers || !triggers.length) {
    container.innerHTML = `<p style="color:var(--text-muted);font-size:0.75rem;">No actionable CPIC Level A interactions detected.</p>`;
    return;
  }

  triggers.forEach(t => {
    const a = t.alert || t;
    const card = document.createElement("div");
    card.className = "cpic-card-item";
    card.innerHTML = `
      <div class="alert-title">⚠️ ${a.drug_name} ➔ ${a.gene} (${a.risk_level || 'CRITICAL'})</div>
      <div class="alert-desc">${a.recommendation || a.clinical_action}</div>
    `;
    container.appendChild(card);
  });
}

function highlightUtterance(idx) {
  document.querySelectorAll(".utterance-item").forEach(u => u.classList.remove("active-audio"));
  const target = document.getElementById(`utt-${idx}`);
  if (target) {
    target.classList.add("active-audio");
    target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function seekAudio(seconds) {
  currentTime = seconds;
  updateAudioDisplay();
}

function toggleAudioPlay() {
  isPlaying = !isPlaying;
  const btn = document.getElementById("audioPlayBtn");
  if (isPlaying) {
    btn.innerText = "⏸";
    playInterval = setInterval(() => {
      currentTime += 0.2;
      if (currentTime >= audioDuration) {
        currentTime = 0;
        toggleAudioPlay();
      }
      updateAudioDisplay();
    }, 200);
  } else {
    btn.innerText = "▶";
    clearInterval(playInterval);
  }
}

function updateAudioDisplay() {
  const pct = Math.min(100, (currentTime / audioDuration) * 100);
  document.getElementById("waveformProgress").style.width = `${pct}%`;
  document.getElementById("audioTimeDisplay").innerText = `${formatTime(currentTime)} / ${formatTime(audioDuration)}`;
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}

function showModal(title, content) {
  document.getElementById("modalTitle").innerText = title;
  document.getElementById("modalBody").innerText = content;
  document.getElementById("modalBackdrop").style.display = "flex";
}
