/**
 * Gotham Investigation Hub Logic (Mini-Gotham)
 * Person/Target Investigation Graph, Movement Timeline, Defensible Evidence Matrix,
 * and Human-in-the-Loop Adjudication Review Gate.
 */

let activeIncidentId = "INC-2026-0041";
let activeInvestigationData = null;
let selectedCandidateIndex = 0;

// Initialize Investigation Hub
document.addEventListener("DOMContentLoaded", () => {
    loadInvestigationIncident();
});

async function loadInvestigationIncident() {
    try {
        const resp = await fetch("/api/investigation/incidents");
        const data = await resp.json();
        if (data.incidents && data.incidents.length > 0) {
            // Find INC-2026-0041 or default to first
            const gothamInc = data.incidents.find(i => i.case_number === "INC-2026-0041") || data.incidents[0];
            activeIncidentId = gothamInc.incident_id;
            renderIncidentHeader(gothamInc);
            // Run initial find person query
            executeFindPerson();
        }
    } catch (err) {
        console.error("Error loading investigation incident:", err);
    }
}

function renderIncidentHeader(inc) {
    const titleEl = document.getElementById("invCaseNumberTitle");
    if (titleEl) titleEl.innerText = inc.case_number || "INC-2026-0041";

    const badgeEl = document.getElementById("invPriorityBadge");
    if (badgeEl) badgeEl.innerText = `${inc.priority || "CRITICAL"} PRIORITY`;

    const officerEl = document.getElementById("invOfficerBadge");
    if (officerEl) officerEl.innerText = `OFFICER: ${inc.officer_in_charge || "AP-EG-8821"}`;
}

async function executeFindPerson() {
    const btn = document.getElementById("btnFindPerson");
    if (btn) {
        btn.innerHTML = `<span class="pulse-dot"></span> SEARCHING OBSERVATION DATABASE...`;
        btn.disabled = true;
    }

    try {
        const resp = await fetch("/api/investigation/find_person", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                incident_id: activeIncidentId,
                probe_track_id: "481"
            })
        });

        const data = await resp.json();
        if (data.status === "SUCCESS") {
            activeInvestigationData = data;
            renderProbeCard(data.probe);
            renderMovementTimeline(data.movement_timeline);
            renderAssociatedObservations(data.candidate_associations);
            renderEvidenceMatrix(data.candidate_associations[0]);
            renderInvestigationGraph(data.investigation_graph);
        }
    } catch (err) {
        console.error("Error executing person search:", err);
    } finally {
        if (btn) {
            btn.innerHTML = `🔍 FIND THIS PERSON ACROSS CCTV NETWORK`;
            btn.disabled = false;
        }
    }
}

function renderProbeCard(probe) {
    if (!probe) return;
    const faceBadge = document.getElementById("probeFaceBadge");
    if (faceBadge) {
        const isAvail = probe.face_status !== "UNAVAILABLE";
        faceBadge.className = isAvail ? "inv-pill pill-green" : "inv-pill pill-amber";
        faceBadge.innerText = `Face: ${probe.face_status.toLowerCase()}`;
    }

    const bodyBadge = document.getElementById("probeBodyBadge");
    if (bodyBadge) bodyBadge.innerText = `Body: available (OSNet 512-d)`;

    const gaitBadge = document.getElementById("probeGaitBadge");
    if (gaitBadge) gaitBadge.innerText = `Gait: available (1.8 steps/s)`;

    const clothingBadge = document.getElementById("probeClothingBadge");
    if (clothingBadge) clothingBadge.innerText = `Clothing: available (${probe.clothing_upper} / ${probe.clothing_lower})`;

    const heightBadge = document.getElementById("probeHeightBadge");
    if (heightBadge) heightBadge.innerText = `Height: ${probe.estimated_height_cm} cm (±3cm)`;

    const objectBadge = document.getElementById("probeObjectBadge");
    if (objectBadge) {
        const objs = probe.carried_objects || ["backpack"];
        objectBadge.innerText = `Accessories: ${objs.join(", ")} detected`;
    }

    const dirBadge = document.getElementById("probeDirectionBadge");
    if (dirBadge) dirBadge.innerText = `Direction: ${probe.direction || "NORTH"}`;
}

function renderMovementTimeline(timeline) {
    const container = document.getElementById("movementTimelineReel");
    if (!container || !timeline) return;

    container.innerHTML = "";
    timeline.forEach((step, idx) => {
        const isOrigin = step.step_index === 0;
        const stepDiv = document.createElement("div");
        stepDiv.className = `timeline-node ${isOrigin ? "origin" : ""}`;
        
        stepDiv.innerHTML = `
            <div class="node-header">
                <span class="node-camera">${step.camera_id}</span>
                <span class="node-time">${step.time_display}</span>
            </div>
            <div class="node-track">Track ${step.track_id}</div>
            <div class="node-status ${step.is_feasible ? 'feasible' : 'infeasible'}">
                ${isOrigin ? 'INCIDENT ORIGIN' : (step.is_feasible ? '✓ TRANSIT FEASIBLE' : '⚠ VELOCITY CHECK')}
            </div>
            <div class="node-transit-info">${step.summary}</div>
        `;

        container.appendChild(stepDiv);

        // Add connecting pathway if not last
        if (idx < timeline.length - 1) {
            const nextStep = timeline[idx + 1];
            const lineDiv = document.createElement("div");
            lineDiv.className = "timeline-connector";
            lineDiv.innerHTML = `
                <div class="connector-line"></div>
                <div class="connector-badge">${nextStep.transit_distance_m.toFixed(0)}m • ${nextStep.transit_speed_mps.toFixed(1)}m/s</div>
            `;
            container.appendChild(lineDiv);
        }
    });
}

function renderAssociatedObservations(associations) {
    const container = document.getElementById("associatedObservationsGrid");
    if (!container || !associations) return;

    container.innerHTML = "";

    // Include Probe Track 481 as the primary anchor
    const probeCard = document.createElement("div");
    probeCard.className = `assoc-card ${selectedCandidateIndex === -1 ? 'active' : ''}`;
    probeCard.onclick = () => selectCandidateTrack(-1);
    probeCard.innerHTML = `
        <div class="assoc-card-badge probe">SEED OBSERVATION</div>
        <div class="assoc-card-cam">CAM-017 • 18:42:11</div>
        <div class="assoc-card-id">Track 481</div>
        <div class="assoc-card-meta">
            <span>Face: UNAVAILABLE</span>
            <span>Backpack: DETECTED</span>
            <span>Heading: NORTH</span>
        </div>
        <div class="assoc-card-status">ORIGIN PROBE</div>
    `;
    container.appendChild(probeCard);

    associations.forEach((assoc, idx) => {
        const isSel = idx === selectedCandidateIndex;
        const card = document.createElement("div");
        card.className = `assoc-card ${isSel ? 'active' : ''}`;
        card.onclick = () => selectCandidateTrack(idx);

        const candTimeStr = assoc.candidate_time > 1000 
            ? new Date(assoc.candidate_time * 1000).toLocaleTimeString([], { hour12: false }) 
            : `18:${49 + idx * 7}:00`;

        card.innerHTML = `
            <div class="assoc-card-badge candidate">CANDIDATE ASSOCIATION</div>
            <div class="assoc-card-cam">${assoc.candidate_camera} • ${candTimeStr}</div>
            <div class="assoc-card-id">Track ${assoc.candidate_track_id}</div>
            <div class="assoc-card-meta">
                <span>Body Sim: ${(assoc.evidence_breakdown.body.score * 100).toFixed(0)}%</span>
                <span>Gait: ${(assoc.evidence_breakdown.gait.score * 100).toFixed(0)}%</span>
                <span>Travel: ${assoc.transit_speed_mps.toFixed(1)} m/s</span>
            </div>
            <div class="assoc-card-status ${assoc.association_status.toLowerCase()}">${assoc.association_status}</div>
        `;
        container.appendChild(card);
    });
}

function selectCandidateTrack(idx) {
    selectedCandidateIndex = idx;
    if (activeInvestigationData && activeInvestigationData.candidate_associations) {
        renderAssociatedObservations(activeInvestigationData.candidate_associations);
        if (idx >= 0 && idx < activeInvestigationData.candidate_associations.length) {
            renderEvidenceMatrix(activeInvestigationData.candidate_associations[idx]);
        }
    }
}

function renderEvidenceMatrix(assoc) {
    if (!assoc) return;
    const ev = assoc.evidence_breakdown || {};

    const container = document.getElementById("evidenceMatrixList");
    if (!container) return;

    const rows = [
        {
            modality: "FACE RECOGNITION",
            status: ev.face?.available ? "AVAILABLE" : "UNAVAILABLE",
            grade: ev.face?.grade || "UNAVAILABLE",
            badgeClass: ev.face?.available ? "badge-green" : "badge-amber",
            description: ev.face?.summary || "Face unavailable (subject turned away / masked)"
        },
        {
            modality: "BODY APPEARANCE (OSNet 512-d)",
            status: "AVAILABLE",
            grade: ev.body?.grade || "STRONG",
            badgeClass: "badge-cyan",
            description: ev.body?.summary || "Strong evidence (OSNet Re-ID cosine similarity)"
        },
        {
            modality: "GAIT DYNAMICS & WAVEFORM",
            status: "AVAILABLE",
            grade: ev.gait?.grade || "MODERATE",
            badgeClass: "badge-emerald",
            description: ev.gait?.summary || "Moderate evidence (Cadence 1.8 steps/s, stride consistency)"
        },
        {
            modality: "CLOTHING CONSISTENCY",
            status: "AVAILABLE",
            grade: ev.clothing?.grade || "SUPPORTING",
            badgeClass: "badge-blue",
            description: ev.clothing?.summary || "Clothing supporting evidence (Upper #1B2430, Lower #2C3539)"
        },
        {
            modality: "SPATIO-TEMPORAL TRAJECTORY",
            status: "CONSISTENT",
            grade: "CONSISTENT",
            badgeClass: "badge-teal",
            description: `Consistent progression along northward corridor towards ${assoc.candidate_camera}`
        },
        {
            modality: "TRAVEL TIME & VELOCITY",
            status: "CONSISTENT",
            grade: "CONSISTENT",
            badgeClass: "badge-green",
            description: ev.travel_time?.summary || `${assoc.time_delta_sec/60.0:.1f} mins for ${assoc.distance_meters:.0f}m (${assoc.transit_speed_mps:.1f} m/s)`
        },
        {
            modality: "CARRIED ACCESSORIES",
            status: "DETECTED",
            grade: ev.carried_objects?.grade || "STRONG",
            badgeClass: "badge-purple",
            description: ev.carried_objects?.summary || "Backpack carried consistently across cameras"
        }
    ];

    container.innerHTML = rows.map(r => `
        <div class="evidence-matrix-row">
            <div class="evidence-modality-col">
                <strong>${r.modality}</strong>
                <span class="evidence-pill ${r.badgeClass}">${r.grade}</span>
            </div>
            <div class="evidence-desc-col">
                ${r.description}
            </div>
        </div>
    `).join("");

    // Update target track in review button
    const revBtn = document.getElementById("btnReviewAssociation");
    if (revBtn) {
        revBtn.onclick = () => openInvestigationReviewModal(assoc);
    }
}

function renderInvestigationGraph(graphData) {
    const canvas = document.getElementById("investigationGraphCanvas");
    if (!canvas || !graphData) return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width = canvas.parentElement.clientWidth || 800;
    const h = canvas.height = 320;

    ctx.clearRect(0, 0, w, h);

    const nodes = graphData.nodes || [];
    const edges = graphData.edges || [];

    // Simple deterministic layered layout
    const nodePositions = {};
    const layers = { "INCIDENT": 60, "TRACK": 160, "CAMERA": 260 };

    // Group tracks and cameras horizontally
    const tracks = nodes.filter(n => n.type === "TRACK");
    const cameras = nodes.filter(n => n.type === "CAMERA");

    // Incident top-center
    const incNode = nodes.find(n => n.type === "INCIDENT");
    if (incNode) {
        nodePositions[incNode.id] = { x: w / 2, y: layers["INCIDENT"] };
    }

    // Tracks middle row
    tracks.forEach((t, i) => {
        const x = (w / (tracks.length + 1)) * (i + 1);
        nodePositions[t.id] = { x, y: layers["TRACK"] };
    });

    // Cameras bottom row
    cameras.forEach((c, i) => {
        const x = (w / (cameras.length + 1)) * (i + 1);
        nodePositions[c.id] = { x, y: layers["CAMERA"] };
    });

    // Draw Edges
    edges.forEach(e => {
        const p1 = nodePositions[e.source];
        const p2 = nodePositions[e.target];
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = e.type === "CONFIRMED_LINK" ? "#10b981" : (e.type === "CANDIDATE_LINK" ? "#06b6d4" : "rgba(255,255,255,0.15)");
            ctx.lineWidth = e.type === "CONFIRMED_LINK" ? 2.5 : 1.5;
            if (e.type === "CANDIDATE_LINK") ctx.setLineDash([4, 4]);
            else ctx.setLineDash([]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    // Draw Nodes
    nodes.forEach(n => {
        const pos = nodePositions[n.id];
        if (!pos) return;

        ctx.beginPath();
        let color = "#3b82f6";
        let r = 16;
        if (n.type === "INCIDENT") { color = "#ef4444"; r = 20; }
        else if (n.type === "TRACK") { color = "#06b6d4"; r = 16; }
        else if (n.type === "CAMERA") { color = "#8b5cf6"; r = 14; }

        ctx.arc(pos.x, pos.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "11px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(n.label, pos.x, pos.y + r + 13);
    });
}

function openInvestigationReviewModal(assoc) {
    const modal = document.getElementById("investigationReviewModal");
    if (!modal || !assoc) return;

    const summaryEl = document.getElementById("invModalCandidateSummary");
    if (summaryEl) {
        summaryEl.innerHTML = `
            <div style="background:#151d2f; padding:12px; border-radius:6px; margin-bottom:14px; border:1px solid #2a3a5c;">
                <div style="color:var(--accent-cyan); font-weight:700; margin-bottom:4px;">
                    Candidate Association: Probe Track ${assoc.probe_track_id} ↔ Candidate Track ${assoc.candidate_track_id}
                </div>
                <div style="font-size:0.82rem; color:var(--text-secondary)">
                    Origin: ${assoc.probe_camera} ➔ Transit To: ${assoc.candidate_camera} (${assoc.distance_meters}m in ${(assoc.time_delta_sec/60).toFixed(1)} mins)
                </div>
                <div style="font-size:0.80rem; color:var(--text-muted); margin-top:6px;">
                    ${assoc.qualitative_narrative}
                </div>
            </div>
        `;
    }

    modal.dataset.relationshipId = `REL-${assoc.probe_track_id}-${assoc.candidate_track_id}`;
    modal.dataset.targetId = assoc.candidate_track_id;
    modal.classList.add("active");
}

function closeInvestigationReviewModal() {
    const modal = document.getElementById("investigationReviewModal");
    if (modal) modal.classList.remove("active");
}

async function submitAdjudicationDecision(decision) {
    const modal = document.getElementById("investigationReviewModal");
    const relId = modal?.dataset.relationshipId || "REL-481-774";
    const targetId = modal?.dataset.targetId || "774";

    const badge = document.getElementById("invReviewOfficerBadge")?.value || "AP-EG-8821";
    const name = document.getElementById("invReviewOfficerName")?.value || "Inspector R. Varma";
    const notes = document.getElementById("invReviewNotesInput")?.value || "Adjudicated candidate transit link.";

    try {
        const resp = await fetch("/api/investigation/review", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                relationship_id: relId,
                reviewer_badge: badge,
                reviewer_name: name,
                decision: decision,
                review_notes: notes,
                target_id: targetId
            })
        });

        const data = await resp.json();
        if (data.status === "SUCCESS") {
            closeInvestigationReviewModal();
            // Re-run search to update statuses
            executeFindPerson();
            alert(`Adjudication Recorded in Tamper-Evident Ledger: ${decision} (${data.adjudication.status})`);
        }
    } catch (err) {
        console.error("Error submitting review:", err);
    }
}

async function reseedGothamDemo() {
    try {
        const resp = await fetch("/api/investigation/seed_demo", { method: "POST" });
        const data = await resp.json();
        if (data.status === "SUCCESS") {
            loadInvestigationIncident();
        }
    } catch (err) {
        console.error("Error reseeding demo:", err);
    }
}
