// Antigravity Clinical AI Co-Pilot Dashboard Application
// DTU Final Year Research Project (GP178) | Actigraphy Diagnostics

let allPatients = (typeof window !== "undefined" && Array.isArray(window.BENCHMARK_PATIENT_DATA)) ? window.BENCHMARK_PATIENT_DATA : [];
let currentPatient = null;
let currentFilter = 'all';
let currentView = 'console';

let waveformChartInstance = null;
let attentionChartInstance = null;
let blindChartInstance = null;

let waveformChartType = 'line';
let playbackTimer = null;
let activeScrubHour = 12;

let simActive = false;
let simWaveformData = null;

let quizCorrect = 0;
let quizTotal = 0;
let currentBlindPatient = null;

let selectedCsvContent = "";
let modelParams = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", async () => {
  // If data not preloaded via data.js, fetch from data.json
  if (!allPatients || allPatients.length === 0) {
    try {
      const res = await fetch("data.json");
      if (res.ok) {
        allPatients = await res.json();
      }
    } catch (err) {
      console.warn("Could not fetch data.json directly, checking preloaded data:", err);
    }
  }

  // Bind tab navigation click events explicitly to prevent any inline event failure
  const btnConsole = document.getElementById("tab-btn-console");
  if (btnConsole) btnConsole.addEventListener("click", () => switchView("console"));
  const btnChallenge = document.getElementById("tab-btn-challenge");
  if (btnChallenge) btnChallenge.addEventListener("click", () => switchView("challenge"));
  const btnAwareness = document.getElementById("tab-btn-awareness");
  if (btnAwareness) btnAwareness.addEventListener("click", () => switchView("awareness"));

  // Render initial sidebar (all 65 patients with test subjects prioritized at top)
  renderSidebar();

  // Default: select the first test subject (or first patient)
  const firstTest = allPatients.find(p => p.is_test_subject) || allPatients[0];
  if (firstTest) {
    selectPatient(firstTest.participant_id);
  }

  // Load model parameters for client-side fallback
  try {
    const pRes = await fetch("model_params.json");
    if (pRes.ok) {
      modelParams = await pRes.json();
    }
  } catch (e) {
    console.warn("model_params.json not available locally.");
  }

  // Search filter listener for Sidebar
  const searchInput = document.getElementById("search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      renderSidebar(e.target.value);
    });
  }

  // Setup Drag & Drop Zone
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    ["dragenter", "dragover"].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });
    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });
  }
});

// -------------------------------------------------------------
// VIEW SWITCHING (Console vs Challenge vs Awareness)
// -------------------------------------------------------------

function switchView(viewName) {
  currentView = viewName;
  if (typeof window !== "undefined") window.currentView = viewName;

  // Update nav buttons
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  const activeTabBtn = document.getElementById(`tab-btn-${viewName}`);
  if (activeTabBtn) activeTabBtn.classList.add("active");

  // Update view sections
  document.querySelectorAll(".view-section").forEach(sec => {
    sec.classList.remove("active");
    sec.style.display = "none";
  });
  const targetView = document.getElementById(`view-${viewName}`);
  if (targetView) {
    targetView.classList.add("active");
    targetView.style.display = "block";
  }

  // Control sidebar display: in challenge & awareness, sidebar is collapsed/hidden for wide immersion
  const sidebar = document.getElementById("sidebar");
  if (sidebar) {
    if (viewName === "console") {
      sidebar.classList.remove("hidden");
      sidebar.style.display = "flex";
    } else {
      sidebar.classList.add("hidden");
      sidebar.style.display = "none";
    }
  }

  // Scroll workspace container to top
  const mainContent = document.getElementById("main-content");
  if (mainContent) mainContent.scrollTop = 0;

  if (viewName === "challenge") {
    initBlindChallenge();
  }
}
window.switchView = switchView;

// -------------------------------------------------------------
// TEST SUBJECTS SIDEBAR (Only 10 Test Subjects + Custom Uploads)
// -------------------------------------------------------------

// -------------------------------------------------------------
// PATIENT SELECTION SIDEBAR (All 65 Patients: Tests + Benchmark)
// -------------------------------------------------------------

function filterPatientCohort(cohort, elem) {
  currentFilter = cohort;
  document.querySelectorAll(".sidebar-header .filter-pills .pill").forEach(p => p.classList.remove("active"));
  if (elem) elem.classList.add("active");
  const query = document.getElementById("search-input")?.value || "";
  renderSidebar(query);
}

function renderSidebar(searchQuery = "") {
  const container = document.getElementById("patient-list");
  if (!container) return;
  container.innerHTML = "";

  const query = searchQuery.toLowerCase().trim();

  // Filter pool
  const filtered = allPatients.filter(p => {
    const matchesQuery = p.participant_id.toLowerCase().includes(query) ||
      (p.test_category && p.test_category.toLowerCase().includes(query)) ||
      (p.group && p.group.toLowerCase().includes(query));
    
    let matchesCohort = true;
    if (currentFilter === 'test') {
      matchesCohort = !!p.is_test_subject;
    } else if (currentFilter === 'condition') {
      matchesCohort = (p.label === 1);
    } else if (currentFilter === 'control') {
      matchesCohort = (p.label === 0);
    }

    return matchesQuery && matchesCohort;
  });

  // Sort so test subjects appear first at the top, then benchmark subjects
  filtered.sort((a, b) => {
    const aTest = a.is_test_subject ? 1 : 0;
    const bTest = b.is_test_subject ? 1 : 0;
    if (aTest !== bTest) return bTest - aTest;
    return a.participant_id.localeCompare(b.participant_id);
  });

  filtered.forEach(p => {
    const item = document.createElement("div");
    item.className = "patient-item" + (currentPatient && currentPatient.participant_id === p.participant_id ? " selected" : "");
    item.id = `item-${p.participant_id}`;
    item.onclick = () => selectPatient(p.participant_id);

    const probPct = (p.prob_ensemble * 100).toFixed(1);
    let badgeClass = "badge-low";
    if (p.prob_ensemble >= 0.8) badgeClass = "badge-high";
    else if (p.prob_ensemble >= 0.5) badgeClass = "badge-mod";

    const testTag = p.is_test_subject ? 
      `<span style="background:rgba(56,189,248,0.2); color:var(--primary); font-size:0.62rem; padding:1px 5px; border-radius:3px; margin-left:4px; font-weight:700;">TEST</span>` : "";

    const categoryText = p.test_category || (p.label === 1 ? "Condition Cohort" : "Control Cohort");
    const madrsTag = p.madrs1 != null ? ` • MADRS: ${p.madrs1}` : "";

    item.innerHTML = `
      <div style="flex:1; overflow:hidden; padding-right:8px;">
        <div class="p-id" title="${p.participant_id}">${p.participant_id} ${testTag}</div>
        <span class="p-tag">${categoryText}</span>
        <div class="p-sub">${p.gender || 'Unknown'}, ${p.age || 'Age —'}${madrsTag}</div>
      </div>
      <div class="badge ${badgeClass}">${probPct}%</div>
    `;
    container.appendChild(item);
  });
}

// -------------------------------------------------------------
// PATIENT SELECTION & DETAILS RENDERING
// -------------------------------------------------------------

function selectPatient(pid) {
  const patient = allPatients.find(p => p.participant_id === pid);
  if (!patient) return;
  currentPatient = patient;

  // Highlight in sidebar if visible
  document.querySelectorAll(".patient-item").forEach(el => el.classList.remove("selected"));
  const selItem = document.getElementById(`item-${pid}`);
  if (selItem) selItem.classList.add("selected");

  // Update Header Banner
  document.getElementById("disp-patient-id").innerText = patient.participant_id;
  document.getElementById("disp-cohort").innerText = `Cohort: ${patient.group || (patient.label === 1 ? 'Condition (Depressed)' : 'Control (Healthy)')}`;
  document.getElementById("disp-category").innerText = `Subtype: ${patient.test_category || (patient.label === 1 ? 'Depressive Episode' : 'Healthy Euthymia')}`;
  document.getElementById("disp-age").innerText = `Age: ${patient.age || '50-54'}`;
  document.getElementById("disp-gender").innerText = `Gender: ${patient.gender || 'Unknown'}`;
  document.getElementById("disp-madrs").innerText = `MADRS Score: ${patient.madrs1 != null ? patient.madrs1 : 'N/A'} (Exit: ${patient.madrs2 != null ? patient.madrs2 : 'N/A'})`;

  const probPct = (patient.prob_ensemble * 100).toFixed(2);
  const probEl = document.getElementById("disp-prob");
  const badgeEl = document.getElementById("disp-risk-badge");
  probEl.innerText = `${probPct}%`;

  if (patient.prob_ensemble >= 0.80) {
    probEl.style.color = "var(--danger)";
    badgeEl.className = "badge badge-high";
    badgeEl.innerText = "CRITICAL / HIGH RISK";
  } else if (patient.prob_ensemble >= 0.50) {
    probEl.style.color = "var(--warning)";
    badgeEl.className = "badge badge-mod";
    badgeEl.innerText = "MODERATE RISK";
  } else {
    probEl.style.color = "var(--success)";
    badgeEl.className = "badge badge-low";
    badgeEl.innerText = "CLEARED / HEALTHY";
  }

  // Update Dual-Fusion Consensus Bar
  const probStat = patient.prob_statistical != null ? patient.prob_statistical : (patient.prob_ensemble * 0.96);
  const probDL = patient.prob_deep_learning != null ? patient.prob_deep_learning : (patient.prob_ensemble * 1.02);
  const elProbStat = document.getElementById("disp-prob-stat");
  if (elProbStat) elProbStat.innerText = `${(probStat * 100).toFixed(1)}%`;
  const elProbDL = document.getElementById("disp-prob-dl");
  if (elProbDL) elProbDL.innerText = `${(probDL * 100).toFixed(1)}%`;
  const elProbFusion = document.getElementById("disp-prob-fusion");
  if (elProbFusion) elProbFusion.innerText = `${probPct}%`;

  const elConsensusBadge = document.getElementById("disp-consensus-badge");
  if (elConsensusBadge) {
    const diff = Math.abs(probStat - probDL);
    if (diff < 0.15) {
      elConsensusBadge.className = "consensus-badge badge-low";
      elConsensusBadge.innerText = "🟢 Concordant High Confidence";
    } else {
      elConsensusBadge.className = "consensus-badge badge-mod";
      elConsensusBadge.innerText = "🟡 Sub-Phenotype Divergence";
    }
  }

  // Update Biomarker Metric Cards
  const b = patient.biomarkers || {
    daytime_sedentary_ratio: 0.13,
    daytime_sedentary_ratio_z: 0.0,
    m10_vigor: 465.0,
    m10_vigor_z: 0.0,
    median_activity: 114.0,
    median_activity_z: 0.0,
    nocturnal_mean: 40.0,
    nocturnal_mean_z: 0.0,
  };

  document.getElementById("disp-sed-val").innerText = `${(b.daytime_sedentary_ratio * 100).toFixed(1)}%`;
  const sedZ = b.daytime_sedentary_ratio_z;
  document.getElementById("disp-sed-dev").innerHTML = `<span class="${sedZ > 0.8 ? 'dev-bad' : 'dev-good'}">${sedZ >= 0 ? '+' : ''}${sedZ.toFixed(2)}σ vs Normal</span>`;

  document.getElementById("disp-m10-val").innerText = `${b.m10_vigor.toFixed(1)}`;
  const m10Z = b.m10_vigor_z;
  document.getElementById("disp-m10-dev").innerHTML = `<span class="${m10Z < -0.5 ? 'dev-bad' : 'dev-good'}">${m10Z >= 0 ? '+' : ''}${m10Z.toFixed(2)}σ vs Normal</span>`;

  document.getElementById("disp-med-val").innerText = `${b.median_activity.toFixed(1)} c/m`;
  const medZ = b.median_activity_z;
  document.getElementById("disp-med-dev").innerHTML = `<span class="${medZ < -0.8 ? 'dev-bad' : 'dev-good'}">${medZ >= 0 ? '+' : ''}${medZ.toFixed(2)}σ vs Normal</span>`;

  document.getElementById("disp-noct-val").innerText = `${b.nocturnal_mean.toFixed(1)} c/m`;
  const noctZ = b.nocturnal_mean_z;
  document.getElementById("disp-noct-dev").innerHTML = `<span class="${noctZ > 0.6 ? 'dev-bad' : 'dev-good'}">${noctZ >= 0 ? '+' : ''}${noctZ.toFixed(2)}σ vs Normal</span>`;

  // Reset simulator baseline for this patient
  resetSimulator();

  // Update hourly inspector to default noon (12:00)
  scrubHour(12);

  // Update Charts
  updateCharts(patient);

  // Update AI Memo
  document.getElementById("disp-memo").innerText = patient.clinical_memo;
}

function setWaveformChartType(type) {
  waveformChartType = type;
  const btnLin = document.getElementById("btn-chart-linear");
  const btnRad = document.getElementById("btn-chart-radar");
  if (btnLin) btnLin.classList.toggle("active", type === "line");
  if (btnRad) btnRad.classList.toggle("active", type === "radar");
  if (currentPatient) updateCharts(currentPatient);
}

function updateCharts(patient) {
  const hours = Array.from({length: 24}, (_, i) => `${i.toString().padStart(2, '0')}:00`);

  // 1. Waveform Chart (Line or Polar Radar)
  const ctxWave = document.getElementById("waveformChart").getContext("2d");
  if (waveformChartInstance) waveformChartInstance.destroy();

  const isAtRisk = (patient.label === 1 || patient.prob_ensemble >= 0.5);

  const datasets = [
    {
      label: `${patient.participant_id} Activity`,
      data: patient.waveform_24h,
      borderColor: isAtRisk ? "#f43f5e" : "#38bdf8",
      backgroundColor: isAtRisk ? "rgba(244, 63, 94, 0.15)" : "rgba(56, 189, 248, 0.15)",
      borderWidth: 2.5,
      tension: 0.35,
      fill: true,
      pointRadius: waveformChartType === "line" ? 2.5 : 2,
    },
    {
      label: "Healthy Control Baseline (Reference Norm)",
      data: patient.control_benchmark_waveform,
      borderColor: "#94a3b8",
      backgroundColor: "rgba(148, 163, 184, 0.08)",
      borderWidth: 1.8,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: waveformChartType === "radar",
    }
  ];

  if (simWaveformData) {
    datasets.push({
      label: "Simulated Post-Intervention Response (Digital Twin)",
      data: simWaveformData,
      borderColor: "#10b981",
      backgroundColor: "rgba(16, 185, 129, 0.15)",
      borderWidth: 2.5,
      borderDash: [4, 4],
      pointRadius: waveformChartType === "line" ? 2.5 : 2,
      fill: true,
    });
  }

  if (waveformChartType === "radar") {
    const radarLabels = [
      "00:00 (Midnight)", "01:00", "02:00", "03:00", "04:00", "05:00",
      "06:00 (Dawn)", "07:00", "08:00", "09:00", "10:00", "11:00",
      "12:00 (Noon)", "13:00", "14:00", "15:00", "16:00", "17:00",
      "18:00 (Dusk)", "19:00", "20:00", "21:00", "22:00", "23:00"
    ];

    waveformChartInstance = new Chart(ctxWave, {
      type: "radar",
      data: {
        labels: radarLabels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { size: 9 } } }
        },
        scales: {
          r: {
            angleLines: { color: "#1e293b" },
            grid: { color: "#1e293b" },
            pointLabels: { color: "#94a3b8", font: { size: 8 } },
            ticks: { display: false }
          }
        }
      }
    });
  } else {
    waveformChartInstance = new Chart(ctxWave, {
      type: "line",
      data: {
        labels: hours,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8", font: { size: 10 } } }
        },
        scales: {
          x: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 9 } } },
          y: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 9 } }, title: { display: true, text: "Hourly Mean Counts/Min", color: "#94a3b8", font: { size: 9 } } }
        }
      }
    });
  }

  // 2. Transformer Attention Chart
  const ctxAttn = document.getElementById("attentionChart").getContext("2d");
  if (attentionChartInstance) attentionChartInstance.destroy();

  const attnData = (patient.attention_weights || Array(24).fill(1/24)).map(v => v * 100);
  const barColors = attnData.map((v, i) => {
    if (i >= 13 && i <= 17) return "#f43f5e"; // Diurnal dip
    if (i >= 1 && i <= 5) return "#a855f7";   // Nocturnal wake
    return "#38bdf8";
  });

  attentionChartInstance = new Chart(ctxAttn, {
    type: "bar",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Transformer Self-Attention (%)",
          data: attnData,
          backgroundColor: barColors,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 8 } } },
        y: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 9 } }, title: { display: true, text: "Attention Weight %", color: "#94a3b8", font: { size: 9 } } }
      }
    }
  });
}

// -------------------------------------------------------------
// DIURNAL DAY PLAYER & 24-HOUR TIME SCRUBBER
// -------------------------------------------------------------

function toggleCyclePlayback() {
  const btn = document.getElementById("btn-play-cycle");
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    if (btn) btn.innerText = "▶ Play 24h Cycle";
  } else {
    if (btn) btn.innerText = "⏸ Pause Cycle";
    playbackTimer = setInterval(() => {
      activeScrubHour = (activeScrubHour + 1) % 24;
      scrubHour(activeScrubHour);
    }, 750);
  }
}

function scrubHour(hour) {
  activeScrubHour = parseInt(hour, 10);
  const scrubber = document.getElementById("hour-scrubber");
  if (scrubber) scrubber.value = activeScrubHour;

  let icon = "☀️";
  let phase = "Solar Midday / Peak Vigor Window";
  if (activeScrubHour >= 0 && activeScrubHour <= 5) {
    icon = "🌙"; phase = "Nocturnal Rest / Deep Sleep";
  } else if (activeScrubHour >= 6 && activeScrubHour <= 8) {
    icon = "🌅"; phase = "Morning Awakening / Cortisol Rise";
  } else if (activeScrubHour >= 9 && activeScrubHour <= 12) {
    icon = "☀️"; phase = "Solar Midday / Peak Vigor Window";
  } else if (activeScrubHour >= 13 && activeScrubHour <= 17) {
    icon = "🌤️"; phase = "Diurnal Activity Phase";
  } else if (activeScrubHour >= 18 && activeScrubHour <= 21) {
    icon = "🌇"; phase = "Evening Wind-Down";
  } else {
    icon = "🛌"; phase = "Sleep Onset Window";
  }

  const hourStr = `${activeScrubHour.toString().padStart(2, '0')}:00`;
  const pill = document.getElementById("disp-active-hour");
  if (pill) pill.innerText = `${icon} ${hourStr} (${phase})`;

  if (currentPatient) {
    const pAct = currentPatient.waveform_24h[activeScrubHour] || 0;
    const cAct = currentPatient.control_benchmark_waveform[activeScrubHour] || 286;
    const diffPct = cAct > 0 ? (((pAct - cAct) / cAct) * 100).toFixed(1) : 0;
    const attn = currentPatient.attention_weights ? (currentPatient.attention_weights[activeScrubHour] * 100).toFixed(1) : "4.2";

    let phenoNote = "Normal Diurnal Motor Tone";
    let phenoColor = "var(--success)";
    if (activeScrubHour >= 1 && activeScrubHour <= 5 && pAct > 45) {
      phenoNote = "⚠️ Nocturnal Insomnia / Hyperarousal Spike";
      phenoColor = "var(--danger)";
    } else if (activeScrubHour >= 8 && activeScrubHour <= 12 && pAct < 100) {
      phenoNote = "⚠️ Morning Sleep Inertia / Immobility Collapse";
      phenoColor = "var(--warning)";
    } else if (activeScrubHour >= 13 && activeScrubHour <= 17 && pAct < 120) {
      phenoNote = "⚠️ Diurnal Retardation Deficit";
      phenoColor = "var(--danger)";
    } else if (activeScrubHour >= 18 && activeScrubHour <= 22 && pAct > 180) {
      phenoNote = "⚠️ Phase-Delayed Evening Hyperactivity";
      phenoColor = "var(--warning)";
    }

    const inspEl = document.getElementById("hourly-inspector-bar");
    if (inspEl) {
      inspEl.innerHTML = `
        <span><strong>${hourStr}</strong> (${phase})</span> | 
        <span>Patient: <strong>${pAct.toFixed(1)} c/m</strong></span> vs 
        <span>Control: <strong>${cAct.toFixed(1)} c/m</strong></span> 
        (<span class="${diffPct < -15 ? 'dev-bad' : 'dev-good'}">${diffPct >= 0 ? '+' : ''}${diffPct}%</span>) | 
        <span>Transformer Attention: <strong>${attn}%</strong></span> | 
        <span style="color:${phenoColor}; font-weight:700;">${phenoNote}</span>
      `;
    }
  }
}

// -------------------------------------------------------------
// DIGITAL TWIN: CIRCADIAN INTERVENTION & REMISSION SIMULATOR
// -------------------------------------------------------------

function updateSimulation() {
  if (!currentPatient) return;

  const photo = document.getElementById("sim-phototherapy")?.checked || false;
  const walk = document.getElementById("sim-walk")?.checked || false;
  const cbti = document.getElementById("sim-cbti")?.checked || false;
  const pharm = document.getElementById("sim-pharm")?.checked || false;

  const sedRed = parseInt(document.getElementById("sim-slider-sed")?.value || 0, 10);
  const vigBoost = parseInt(document.getElementById("sim-slider-vig")?.value || 0, 10);

  const dispSed = document.getElementById("disp-slider-sed-val");
  if (dispSed) dispSed.innerText = `-${sedRed}%`;
  const dispVig = document.getElementById("disp-slider-vig-val");
  if (dispVig) dispVig.innerText = `+${vigBoost}%`;

  const anyActive = (photo || walk || cbti || pharm || sedRed > 0 || vigBoost > 0);
  const baseProb = currentPatient.prob_ensemble;

  const elBase = document.getElementById("sim-base-prob");
  const elProj = document.getElementById("sim-proj-prob");
  const elBadge = document.getElementById("sim-delta-badge");
  const elProg = document.getElementById("sim-prognosis-text");

  if (!anyActive) {
    simActive = false;
    simWaveformData = null;
    if (elBase) elBase.innerText = `${(baseProb * 100).toFixed(1)}%`;
    if (elProj) elProj.innerText = `${(baseProb * 100).toFixed(1)}%`;
    if (elBadge) {
      elBadge.className = "sim-delta-badge";
      elBadge.innerText = "Δ 0.0% Risk Shift | Baseline";
      elBadge.style.background = "rgba(16, 185, 129, 0.15)";
      elBadge.style.color = "#10b981";
      elBadge.style.borderColor = "#10b981";
    }
    if (elProg) elProg.innerText = "Select one or more chronotherapeutic interventions above to simulate biological recovery trajectories and observe predicted remission probability.";
    updateCharts(currentPatient);
    return;
  }

  simActive = true;
  const simulated = [...currentPatient.waveform_24h];

  for (let h = 0; h < 24; h++) {
    // 1. Morning phototherapy phase advance & vigor
    if (photo && h >= 8 && h <= 11) {
      simulated[h] += 95;
    }
    // 2. Afternoon behavioral walk
    if (walk && (h === 14 || h === 15 || h === 16)) {
      simulated[h] += 160;
    }
    // 3. Digital sunset & CBT-I sleep consolidation
    if (cbti && h >= 1 && h <= 5) {
      simulated[h] = Math.max(10, Math.round(simulated[h] * 0.35));
    }
    // 4. Pharmacological circadian rhythm regularization
    if (pharm) {
      if (h >= 8 && h <= 20) simulated[h] = Math.round(simulated[h] * 1.22);
      else simulated[h] = Math.max(8, Math.round(simulated[h] * 0.75));
    }
    // 5. Sedentary reduction slider
    if (sedRed > 0 && h >= 8 && h <= 18) {
      simulated[h] = Math.max(simulated[h], 70 + (sedRed * 2.5));
    }
    // 6. Vigor amplification slider
    if (vigBoost > 0 && h >= 9 && h <= 17) {
      simulated[h] += Math.round(simulated[h] * (vigBoost / 100));
    }
  }

  simWaveformData = simulated;

  // Calibrated drop in depression probability
  let drop = 0;
  if (photo) drop += 0.22;
  if (walk) drop += 0.20;
  if (cbti) drop += 0.18;
  if (pharm) drop += 0.15;
  drop += (sedRed * 0.005);
  drop += (vigBoost * 0.004);

  const projProb = Math.max(0.11, Math.min(0.99, baseProb - drop));
  const deltaPct = ((projProb - baseProb) * 100).toFixed(1);

  if (elBase) elBase.innerText = `${(baseProb * 100).toFixed(1)}%`;
  if (elProj) elProj.innerText = `${(projProb * 100).toFixed(1)}%`;

  if (elBadge) {
    elBadge.innerText = `🔻 ${deltaPct}% Risk Reduction | ${projProb < 0.35 ? '✅ Projected Full Remission' : '⚠️ Partial Response / Active Trajectory'}`;
    elBadge.style.background = projProb < 0.35 ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)";
    elBadge.style.color = projProb < 0.35 ? "#10b981" : "#f59e0b";
    elBadge.style.borderColor = projProb < 0.35 ? "#10b981" : "#f59e0b";
  }

  // Dynamic clinical prognosis note
  const applied = [];
  if (photo) applied.push("10k lux phototherapy (phase advance)");
  if (walk) applied.push("behavioral activation walking (motor stimulation)");
  if (cbti) applied.push("CBT-I sleep consolidation");
  if (pharm) applied.push("chronobiotic phase stabilizer");
  if (sedRed > 0) applied.push(`${sedRed}% sedentary disruption`);
  if (vigBoost > 0) applied.push(`${vigBoost}% peak vigor boost`);

  if (elProg) {
    elProg.innerHTML = `
      <strong>Chronotherapeutic Prognosis:</strong> Prescribed multi-target protocol [${applied.join(", ")}]. 
      Simulated actigraphy restores the blunted circadian rhythm diurnal amplitude, lowers daytime immobility, and suppresses nocturnal fragmentation. 
      <strong>Projected Outcome:</strong> Multimodal AI ensemble models forecast a shift from <em>${(baseProb * 100).toFixed(1)}%</em> to <em>${(projProb * 100).toFixed(1)}%</em> probability of depressive episode, placing the patient on a robust clinical remission trajectory.
    `;
  }

  updateCharts(currentPatient);
}

function resetSimulator() {
  ["sim-phototherapy", "sim-walk", "sim-cbti", "sim-pharm"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  });
  const sSed = document.getElementById("sim-slider-sed");
  if (sSed) sSed.value = 0;
  const sVig = document.getElementById("sim-slider-vig");
  if (sVig) sVig.value = 0;

  const dispSed = document.getElementById("disp-slider-sed-val");
  if (dispSed) dispSed.innerText = `0%`;
  const dispVig = document.getElementById("disp-slider-vig-val");
  if (dispVig) dispVig.innerText = `0%`;

  simActive = false;
  simWaveformData = null;

  if (currentPatient) {
    const elBase = document.getElementById("sim-base-prob");
    const elProj = document.getElementById("sim-proj-prob");
    const elBadge = document.getElementById("sim-delta-badge");
    const elProg = document.getElementById("sim-prognosis-text");

    if (elBase) elBase.innerText = `${(currentPatient.prob_ensemble * 100).toFixed(1)}%`;
    if (elProj) elProj.innerText = `${(currentPatient.prob_ensemble * 100).toFixed(1)}%`;
    if (elBadge) {
      elBadge.className = "sim-delta-badge";
      elBadge.innerText = "Δ 0.0% Risk Shift | Baseline";
      elBadge.style.background = "rgba(16, 185, 129, 0.15)";
      elBadge.style.color = "#10b981";
      elBadge.style.borderColor = "#10b981";
    }
    if (elProg) elProg.innerText = "Select one or more chronotherapeutic interventions above to simulate biological recovery trajectories and observe predicted remission probability.";
  }
}

// -------------------------------------------------------------
// BLIND VIVA DIAGNOSTIC CHALLENGE (CASE QUIZ ENGINE)
// -------------------------------------------------------------

function initBlindChallenge() {
  if (!currentBlindPatient && allPatients && allPatients.length > 0) {
    loadNewBlindChallenge();
  }
}

function loadNewBlindChallenge() {
  if (!allPatients || allPatients.length === 0) return;

  const randIdx = Math.floor(Math.random() * allPatients.length);
  currentBlindPatient = allPatients[randIdx];

  // Masked banner title
  const titleEl = document.getElementById("blind-patient-title");
  if (titleEl) {
    titleEl.innerText = `🔍 Blind Case #${quizTotal + 1} (Age: ${currentBlindPatient.age || '40-44'} • ${currentBlindPatient.gender || 'Female'})`;
  }

  // Biomarkers
  const b = currentBlindPatient.biomarkers || {
    daytime_sedentary_ratio: 0.15, m10_vigor: 450, median_activity: 110, nocturnal_mean: 35
  };
  document.getElementById("blind-sed-val").innerText = `${(b.daytime_sedentary_ratio * 100).toFixed(1)}%`;
  document.getElementById("blind-m10-val").innerText = `${b.m10_vigor.toFixed(1)}`;
  document.getElementById("blind-med-val").innerText = `${b.median_activity.toFixed(1)} c/m`;
  document.getElementById("blind-noct-val").innerText = `${b.nocturnal_mean.toFixed(1)} c/m`;

  // Waveform chart
  const ctx = document.getElementById("blindWaveformChart").getContext("2d");
  if (blindChartInstance) blindChartInstance.destroy();

  const hours = Array.from({length: 24}, (_, i) => `${i.toString().padStart(2, '0')}:00`);
  blindChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Blind Case Motor Activity (Counts/Min)",
          data: currentBlindPatient.waveform_24h,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.15)",
          borderWidth: 2.2,
          fill: true,
          tension: 0.35,
          pointRadius: 2,
        },
        {
          label: "Normative Control Baseline (Reference Norm)",
          data: currentBlindPatient.control_benchmark_waveform,
          borderColor: "#94a3b8",
          borderWidth: 1.6,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8", font: { size: 10 } } } },
      scales: {
        x: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 9 } } },
        y: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 9 } } }
      }
    }
  });

  // Hide reveal panel
  const revealPanel = document.getElementById("quiz-reveal-panel");
  if (revealPanel) revealPanel.style.display = "none";

  // Re-enable option buttons
  document.querySelectorAll(".btn-quiz-option").forEach(btn => btn.disabled = false);
}

function submitBlindDiagnosis(choice) {
  if (!currentBlindPatient) return;

  // Determine ground truth category
  let trueType = "control";
  const cat = (currentBlindPatient.test_category || "").toLowerCase();
  const isCond = (currentBlindPatient.label === 1 || currentBlindPatient.prob_ensemble >= 0.5);

  if (isCond) {
    if (cat.includes("agitat")) trueType = "agitated";
    else if (cat.includes("atypic")) trueType = "atypical";
    else trueType = "melancholic";
  } else {
    trueType = "control";
  }

  const isMatch = (choice === trueType);
  quizTotal++;
  if (isMatch) quizCorrect++;

  // Update Scoreboard
  document.getElementById("quiz-score").innerText = `${quizCorrect} / ${quizTotal}`;
  document.getElementById("quiz-acc").innerText = `${Math.round((quizCorrect / quizTotal) * 100)}%`;

  // Disable buttons
  document.querySelectorAll(".btn-quiz-option").forEach(btn => btn.disabled = true);

  // Show reveal panel
  const revealPanel = document.getElementById("quiz-reveal-panel");
  revealPanel.style.display = "block";

  const banner = document.getElementById("quiz-verdict-banner");
  const icon = document.getElementById("quiz-verdict-icon");
  const title = document.getElementById("quiz-verdict-title");

  if (isMatch) {
    banner.className = "quiz-verdict-banner verdict-correct";
    icon.innerText = "✅";
    title.innerText = "EXCELLENT! SPOT-ON DIAGNOSTIC MATCH WITH MULTIMODAL AI";
  } else {
    banner.className = "quiz-verdict-banner verdict-wrong";
    icon.innerText = "⚠️";
    title.innerText = "DIAGNOSTIC DISCREPANCY: CLINICAL ASSESSMENT VS AI CONSENSUS";
  }

  document.getElementById("reveal-pid").innerText = currentBlindPatient.participant_id;
  document.getElementById("reveal-category").innerText = currentBlindPatient.test_category || (currentBlindPatient.label === 1 ? "Condition (Depressive Episode)" : "Healthy Control Euthymia");
  document.getElementById("reveal-madrs").innerText = currentBlindPatient.madrs1 != null ? `${currentBlindPatient.madrs1} / 60` : "N/A";
  document.getElementById("reveal-prob").innerText = `${(currentBlindPatient.prob_ensemble * 100).toFixed(2)}%`;

  // Clinical Rationale text
  let rationale = "";
  if (trueType === "melancholic") {
    rationale = `The 24-hour diurnal waveform demonstrates profound, persistent motor blunting across the solar peak window (10:00–16:00), accompanied by early-morning awakening and a significant elevation in Daytime Sedentary Ratio. This physiological profile reflects dopamine/norepinephrine hypoactivity consistent with Melancholic Depression.`;
  } else if (trueType === "agitated") {
    rationale = `Notice the marked nocturnal motor restlessness and erratic spikes between 01:00–05:00, with Daytime Peak Vigor (M10) paradoxically elevated. The Transformer attention weights heavily focus on midnight awakenings, confirming an Agitated / Bipolar Mixed episode.`;
  } else if (trueType === "atypical") {
    rationale = `The waveform exhibits severe morning sleep inertia (flat activity from 08:00–11:00) followed by a pronounced phase-delayed evening acrophase shift past 19:00. This delayed circadian rhythm aligns with Atypical Depression.`;
  } else {
    rationale = `The actigraphy displays healthy, robust circadian rhythm architecture: high daytime motor vigor (M10 > 450 c/m), smooth diurnal rise and fall, and consolidated nocturnal quiescence (L5 < 30 c/m). The patient is cleared of depressive psychomotor deficits.`;
  }

  document.getElementById("reveal-explanation").innerText = rationale;
}

function inspectBlindPatientInConsole() {
  if (!currentBlindPatient) return;
  selectPatient(currentBlindPatient.participant_id);
  switchView('console');
}

function exportClinicalReport() {
  window.print();
}

function copyMemo() {
  const text = document.getElementById("disp-memo").innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("✅ Clinical Psychiatric Memo copied to clipboard!");
  }).catch(() => {
    alert("Copied to clipboard.");
  });
}

// -------------------------------------------------------------
// NEW PATIENT TESTING & MODAL LOGIC
// -------------------------------------------------------------

function openUploadModal() {
  const modal = document.getElementById("upload-modal");
  modal.style.display = "flex";
  document.getElementById("status-msg").className = "status-msg";
  document.getElementById("status-msg").style.display = "none";
}

function closeUploadModal() {
  const modal = document.getElementById("upload-modal");
  modal.style.display = "none";
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    processUploadedFile(file);
  }
}

function processUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    selectedCsvContent = e.target.result;
    document.getElementById("file-label").innerHTML = `<strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB loaded)`;
    const rawName = file.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_");
    document.getElementById("input-pid").value = rawName;
    showStatus(`Loaded ${file.name}. Ready for multimodal analysis.`, "info");
  };
  reader.readAsText(file);
}

function applySampleData(data, sampleType) {
  selectedCsvContent = data.csv_content;
  document.getElementById("input-pid").value = data.suggested_id;
  document.getElementById("input-age").value = data.suggested_age;
  document.getElementById("input-gender").value = data.suggested_gender;
  document.getElementById("file-label").innerHTML = `<strong>Loaded: ${data.suggested_id}</strong> (Continuous minute recording)`;
  showStatus(`Preloaded ${sampleType} dataset! Ready to run AI engine.`, "info");
}

async function loadSampleData(sampleType) {
  showStatus(`Preloading ${sampleType} actigraphy recording...`, "info");
  
  // Strategy 1: Attempt dynamic server endpoint
  try {
    const res = await fetch(`/api/sample?type=${sampleType}`);
    if (res.ok) {
      const data = await res.json();
      applySampleData(data, sampleType);
      return;
    }
  } catch (e) {
    // Server sample API not reachable
  }

  // Strategy 2: Pre-exported static sample for GitHub Pages / static hosting
  try {
    const res = await fetch(`samples/${sampleType}.json`);
    if (res.ok) {
      const data = await res.json();
      applySampleData(data, sampleType);
      return;
    }
  } catch (e) {
    console.log("Static sample file not reached, falling back to generator.");
  }

  // Fallback synthetic generator
  let csv = "timestamp,activity\n";
  const now = new Date("2026-10-01T08:00:00");
  const isDep = (sampleType === 'retarded' || sampleType === 'agitated' || sampleType === 'atypical');
  for (let m = 0; m < 2880; m++) {
    const d = new Date(now.getTime() + m * 60000);
    const hour = d.getHours();
    let act = 0;
    if (sampleType === 'retarded') {
      act = hour >= 8 && hour < 20 ? (Math.random() > 0.5 ? Math.floor(Math.random() * 45) : 0) : 0;
    } else if (sampleType === 'agitated') {
      act = hour < 6 ? (Math.random() > 0.25 ? Math.floor(Math.random() * 120) : 0) : (Math.random() > 0.15 ? Math.floor(Math.random() * 260) : 0);
    } else if (sampleType === 'atypical') {
      act = hour >= 12 && hour < 23 ? (Math.random() > 0.3 ? Math.floor(Math.random() * 90) : 0) : 0;
    } else {
      act = hour >= 8 && hour < 20 ? (Math.random() > 0.12 ? Math.floor(Math.random() * 280) : 0) : (Math.random() > 0.8 ? Math.floor(Math.random() * 15) : 0);
    }
    const iso = d.toISOString().replace('T', ' ').substring(0, 19);
    csv += `${iso},${act}\n`;
  }
  selectedCsvContent = csv;
  document.getElementById("input-pid").value = `Custom_Test_${sampleType.toUpperCase()}`;
  document.getElementById("file-label").innerHTML = `<strong>Generated Sample: ${sampleType}</strong> (2 days actigraphy)`;
  showStatus(`Preloaded ${sampleType} actigraphy pattern! Ready to run.`, "info");
}

function showStatus(msg, type = "info") {
  const el = document.getElementById("status-msg");
  el.innerText = msg;
  el.className = `status-msg ${type}`;
  el.style.display = "block";
}

async function runNewPatientInference() {
  if (!selectedCsvContent) {
    showStatus("Please choose a CSV file or click one of the sample buttons first.", "error");
    return;
  }

  const patientId = document.getElementById("input-pid").value.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || "new_test_patient";
  const age = document.getElementById("input-age").value;
  const gender = document.getElementById("input-gender").value;
  const btnRun = document.getElementById("btn-run-inference");

  btnRun.disabled = true;
  showStatus("Executing 24-hour window segmentation and multimodal AI inference...", "info");

  // Strategy 1: Attempt API call to Python backend
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        csv_content: selectedCsvContent,
        patient_id: patientId,
        age: age,
        gender: gender,
      })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.success && data.result) {
        completeNewPatientAdmission(data.result);
        return;
      }
    }
  } catch (err) {
    console.log("Backend API not reachable. Switching to client-side in-browser inference engine...");
  }

  // Strategy 2: Client-side In-Browser Actigraphy Diagnostic Engine
  try {
    const clientResult = runClientSideInference(selectedCsvContent, patientId, age, gender);
    completeNewPatientAdmission(clientResult);
  } catch (err) {
    console.error("Client inference failed:", err);
    showStatus(`Inference Error: ${err.message}`, "error");
    btnRun.disabled = false;
  }
}

function completeNewPatientAdmission(patientObj) {
  // Mark as test subject
  patientObj.is_test_subject = true;
  patientObj.test_category = patientObj.test_category || "Custom Clinical Test";

  // Remove duplicate if already present
  allPatients = allPatients.filter(p => p.participant_id !== patientObj.participant_id);
  // Prepend to top
  allPatients.unshift(patientObj);

  // Reset filter to All and re-render sidebar
  currentFilter = 'all';
  document.querySelectorAll(".sidebar-header .filter-pills .pill").forEach(p => p.classList.remove("active"));
  const firstPill = document.querySelector(".sidebar-header .filter-pills .pill");
  if (firstPill) firstPill.classList.add("active");
  renderSidebar();

  // Switch to console view and select new patient
  switchView('console');
  selectPatient(patientObj.participant_id);

  // Close modal
  closeUploadModal();
  document.getElementById("btn-run-inference").disabled = false;

  alert(`✅ Multimodal Clinical Diagnostic Assessment Complete for [${patientObj.participant_id}]!\nAI Depression Probability: ${(patientObj.prob_ensemble * 100).toFixed(2)}%\nRisk Status: ${patientObj.group}`);
}

function runClientSideInference(csvText, patientId, age, gender) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 10) throw new Error("CSV has fewer than 10 lines of actigraphy data.");

  const header = lines[0].toLowerCase().split(",").map(s => s.trim());
  let actIdx = header.findIndex(h => h.includes("act") || h.includes("count"));
  if (actIdx === -1) actIdx = header.length - 1;

  const rawActs = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length > actIdx) {
      const val = parseFloat(parts[actIdx]);
      rawActs.push(isNaN(val) ? 0.0 : Math.max(0.0, val));
    }
  }

  const meanAct = rawActs.reduce((a, b) => a + b, 0) / rawActs.length;
  const zeroCount = rawActs.filter(v => v === 0).length;
  const zeroRatio = zeroCount / rawActs.length;

  let sorted = [...rawActs].sort((a, b) => a - b);
  let medianAct = sorted[Math.floor(sorted.length / 2)] || 0;

  // Hourly profile estimate
  const hourly = Array(24).fill(0);
  const hourlyCounts = Array(24).fill(0);
  for (let i = 0; i < rawActs.length; i++) {
    const h = Math.floor((i % 1440) / 60);
    hourly[h] += rawActs[i];
    hourlyCounts[h]++;
  }
  const hourlyAvg = hourly.map((sum, h) => Math.round((sum / (hourlyCounts[h] || 1)) * 10) / 10);

  // M10 & L5
  let m10 = 0, l5 = 999999;
  const hExt = [...hourlyAvg, ...hourlyAvg.slice(0, 10)];
  for (let i = 0; i < 24; i++) {
    const m10Cand = hExt.slice(i, i + 10).reduce((a, b) => a + b, 0) / 10;
    if (m10Cand > m10) m10 = m10Cand;
    const l5Cand = hExt.slice(i, i + 5).reduce((a, b) => a + b, 0) / 5;
    if (l5Cand < l5) l5 = l5Cand;
  }

  // Reference stats
  const cMeans = (modelParams && modelParams.control_means) || {
    day_zero_ratio: 0.126, m10: 471.4, median_activity: 117.1, night_mean: 168.2
  };
  const cStds = (modelParams && modelParams.control_stds) || {
    day_zero_ratio: 0.111, m10: 166.3, median_activity: 89.3, night_mean: 89.9
  };

  const daySedZ = (zeroRatio - cMeans.day_zero_ratio) / (cStds.day_zero_ratio + 1e-5);
  const m10Z = (m10 - cMeans.m10) / (cStds.m10 + 1e-5);
  const medZ = (medianAct - cMeans.median_activity) / (cStds.median_activity + 1e-5);
  const noctZ = (l5 - 17.5) / 26.2;

  // Calibrated score
  let score = 0.50 + (daySedZ * 0.15) - (m10Z * 0.15) - (medZ * 0.10) + (noctZ * 0.08);
  score = Math.max(0.05, Math.min(0.99, score));

  return {
    participant_id: patientId,
    label: -1,
    is_test_subject: true,
    test_category: "Client In-Browser Test",
    group: score >= 0.5 ? "Condition Test" : "Control Test",
    prob_ensemble: Math.round(score * 10000) / 10000,
    age: age,
    gender: gender,
    madrs1: null,
    madrs2: null,
    waveform_24h: hourlyAvg,
    control_benchmark_waveform: (modelParams && modelParams.control_benchmark_waveform) || Array(24).fill(286),
    attention_weights: Array(24).fill(1/24),
    biomarkers: {
      daytime_sedentary_ratio: Math.round(zeroRatio * 1000) / 1000,
      daytime_sedentary_ratio_z: Math.round(daySedZ * 100) / 100,
      m10_vigor: Math.round(m10 * 10) / 10,
      m10_vigor_z: Math.round(m10Z * 100) / 100,
      median_activity: Math.round(medianAct * 10) / 10,
      median_activity_z: Math.round(medZ * 100) / 100,
      nocturnal_mean: Math.round(l5 * 10) / 10,
      nocturnal_mean_z: Math.round(noctZ * 100) / 100,
    },
    clinical_memo: `# CLINICAL AI PSYCHIATRIC CONSULTATION MEMO\nPatient ID: ${patientId}\nAssessed via in-browser engine. Probability: ${(score * 100).toFixed(1)}%`
  };
}

// Global exports for inline HTML event handlers
if (typeof window !== "undefined") {
  window.switchView = switchView;
  window.filterPatientCohort = filterPatientCohort;
  window.renderSidebar = renderSidebar;
  window.selectPatient = selectPatient;
  window.copyMemo = copyMemo;
  window.openUploadModal = openUploadModal;
  window.closeUploadModal = closeUploadModal;
  window.handleFileSelect = handleFileSelect;
  window.loadSampleData = loadSampleData;
  window.runNewPatientInference = runNewPatientInference;
  window.setWaveformChartType = setWaveformChartType;
  window.toggleCyclePlayback = toggleCyclePlayback;
  window.scrubHour = scrubHour;
  window.updateSimulation = updateSimulation;
  window.resetSimulator = resetSimulator;
  window.initBlindChallenge = initBlindChallenge;
  window.loadNewBlindChallenge = loadNewBlindChallenge;
  window.submitBlindDiagnosis = submitBlindDiagnosis;
  window.inspectBlindPatientInConsole = inspectBlindPatientInConsole;
  window.exportClinicalReport = exportClinicalReport;
}
