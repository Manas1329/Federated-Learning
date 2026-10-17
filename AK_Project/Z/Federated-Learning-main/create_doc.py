import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

def create_document():
    doc = Document()

    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Custom Heading 1 Style
    h1_style = doc.styles['Heading 1']
    h1_font = h1_style.font
    h1_font.name = 'Calibri'
    h1_font.size = Pt(18)
    h1_font.bold = True
    h1_font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Deep Navy Blue

    # Custom Heading 2 Style
    h2_style = doc.styles['Heading 2']
    h2_font = h2_style.font
    h2_font.name = 'Calibri'
    h2_font.size = Pt(14)
    h2_font.bold = True
    h2_font.color.rgb = RGBColor(0x25, 0x63, 0xEB) # Royal Blue

    # Custom Heading 3 Style
    h3_style = doc.styles['Heading 3']
    h3_font = h3_style.font
    h3_font.name = 'Calibri'
    h3_font.size = Pt(12)
    h3_font.bold = True
    h3_font.color.rgb = RGBColor(0x0F, 0x76, 0x6E) # Teal

    # Helper functions
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        p.paragraph_format.space_after = Pt(4)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)
        p.paragraph_format.space_after = Pt(24)

    def add_qa_box(q_num, question, answer):
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(12)
        p_q.paragraph_format.space_after = Pt(4)
        run_q_title = p_q.add_run(f"Q{q_num}. {question}")
        run_q_title.bold = True
        run_q_title.font.size = Pt(12)
        run_q_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

        # Answer paragraph formatted as callout box
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.5)

        # Set shading background
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F0F4F8"/>')
        cell._tc.get_or_add_tcPr().append(shd)

        # Left border blue
        borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>')
        cell._tc.get_or_add_tcPr().append(borders)

        p_a = cell.paragraphs[0]
        p_a.paragraph_format.space_before = Pt(6)
        p_a.paragraph_format.space_after = Pt(6)
        p_a.paragraph_format.left_indent = Inches(0.1)
        p_a.paragraph_format.right_indent = Inches(0.1)

        run_a_label = p_a.add_run("Answer: ")
        run_a_label.bold = True
        run_a_label.font.color.rgb = RGBColor(0x1D, 0x4E, 0x89)

        run_a_text = p_a.add_run(answer)
        run_a_text.font.size = Pt(10.5)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Document Header
    add_title("Federated Healthcare AI Platform")
    add_subtitle("5-Layer Technical Architecture & Comprehensive Core Q&A Verification")

    # -------------------------------------------------------------
    # PART 1: 5-LAYER ARCHITECTURAL BREAKDOWN
    # -------------------------------------------------------------
    h1 = doc.add_heading("PART 1: 5-LAYER ARCHITECTURAL BREAKDOWN", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(12)

    # ------------------ LAYER A ------------------
    doc.add_heading("Layer A — Core Machine Learning", level=2)

    doc.add_heading("1. Conceptual Framework & Architecture", level=3)
    p = doc.add_paragraph()
    p.add_run("• Federated vs. Centralized Learning: ").bold = True
    p.add_run("In centralized machine learning, raw medical images from all hospitals are uploaded to a single cloud server for training. In our federated approach, hospitals train the neural network locally on their own hardware. Only model weights/gradients are transmitted to the server coordinator, eliminating the need to centralize patient data.\n")
    p.add_run("• CNN Architecture (ChestCNN in model.py): ").bold = True
    p.add_run("A 2-stage Convolutional Neural Network consisting of:\n")
    p.add_run("   - conv1: 1 -> 32 channels (3x3 kernel, padding 1) + ReLU + MaxPool2d(2x2).\n")
    p.add_run("   - conv2: 32 -> 64 channels (3x3 kernel, padding 1) + ReLU + MaxPool2d(2x2).\n")
    p.add_run("   - fc1: Fully connected layer mapping flattened 64 x 32 x 32 = 65,536 features to 128 hidden units + ReLU.\n")
    p.add_run("   - fc2: Linear layer outputting 2 raw unnormalized logits.\n")
    p.add_run("• Input Pipeline: ").bold = True
    p.add_run("1-channel Grayscale images resized to 128x128 pixels, converted to PyTorch tensors normalized in [0, 1].\n")
    p.add_run("• Classification & Loss Function: ").bold = True
    p.add_run("Binary classification (0 = NORMAL, 1 = PNEUMONIA). The raw outputs are 2-dimensional logits passing into nn.CrossEntropyLoss(), which combines LogSoftmax and NLLLoss. When class imbalance occurs, class weights are dynamically injected into CrossEntropyLoss(weight=weights) via the DACM protocol.")

    doc.add_heading("2. Training, Validation & Metric Diagnostics", level=3)
    p = doc.add_paragraph()
    p.add_run("• Optimization: ").bold = True
    p.add_run("Adam optimizer (lr = 0.001) during local FL rounds; SGD (lr = 0.001) during DACM compensatory fine-tuning.\n")
    p.add_run("• Evaluation Metrics:\n").bold = True
    p.add_run("   - Accuracy: Overall correct predictions ((TP+TN)/Total).\n")
    p.add_run("   - Precision: TP / (TP+FP) — minimizing false alarms.\n")
    p.add_run("   - Recall (Sensitivity): TP / (TP+FN) — critical in medical screening to ensure pneumonia cases are not missed.\n")
    p.add_run("   - F1 Score: Harmonic mean of Precision and Recall.\n")
    p.add_run("   - ROC-AUC & PR-AUC: Threshold-independent diagnostic quality measures computed via sklearn.metrics.")

    doc.add_heading("Likely Interview Questions (Layer A)", level=3)

    p = doc.add_paragraph()
    p.add_run("Likely L1: \"Explain your project.\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"Our project is a privacy-preserving, fault-tolerant Federated Learning platform designed for clinical chest X-ray pneumonia classification across isolated hospital nodes. Instead of centralizing sensitive patient scans, each hospital node trains a local Convolutional Neural Network (ChestCNN) on its own data. The nodes send quantized model parameters to a central server that aggregates them using FedAvg. To handle real-world deployment challenges, the platform integrates Differential Privacy (DP-SGD), dynamic INT8 quantization, an EMA-driven adaptive dropout engine, a non-IID safe client trust scoring module, and a Distribution-Aware Compensation Mechanism (DACM) for post-unlearning recovery.\"\n")

    p.add_run("Likely L2: \"How do you know the model is generalizing rather than overfitting?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"We evaluate generalization through three rigorous practices in evaluate.py and evaluate_comparison.py:\n"
              "1. Data Split Integrity: Each hospital node splits its dataset into an 80% training set and a 20% held-out test set (load_hospital_data in utils.py). Evaluation is performed exclusively on the held-out test set.\n"
              "2. Cross-Hospital Out-of-Distribution Testing: We evaluate the local model trained exclusively on Hospital A against Hospital B and Hospital C test sets (comparison.py). A local model suffers a ~10-15% accuracy drop on out-of-distribution demographics, whereas the aggregated Global FL Model maintains consistent high performance (>= 94% accuracy and F1) across all hospitals.\n"
              "3. Precision vs. Recall Alignment: We track ROC-AUC (0.9803) and PR-AUC (0.9583) alongside confusion matrices to verify that the model learns generalizable pulmonary opacities rather than memorizing local background noise.\"")

    # ------------------ LAYER B ------------------
    doc.add_heading("Layer B — Privacy and Communication", level=2)

    doc.add_heading("1. Differential Privacy (DP-SGD)", level=3)
    p = doc.add_paragraph()
    p.add_run("• Library: ").bold = True
    p.add_run("Integrated using Meta's Opacus library (PrivacyEngine).\n")
    p.add_run("• Mechanism:\n").bold = True
    p.add_run("   1. Per-Sample Gradient Clipping: Clips each individual sample's gradient tensor to a maximum L2 norm threshold (C = DP_MAX_GRAD_NORM = 1.0).\n")
    p.add_run("   2. Noise Injection: Calibrated Gaussian noise (sigma = DP_NOISE_MULTIPLIER = 1.0 or 0.3) is added to the clipped sum before updating weights.\n")
    p.add_run("• Privacy Budget Accounting: ").bold = True
    p.add_run("Privacy loss is tracked as (epsilon, delta)-Differential Privacy, where delta = 10^-5 (probability of privacy breach) and cumulative epsilon is calculated per round via privacy_engine.get_epsilon(delta).")

    doc.add_heading("2. Communication Optimization (Dynamic INT8 Quantization)", level=3)
    p = doc.add_paragraph()
    p.add_run("• Algorithm (quantization.py): ").bold = True
    p.add_run("Symmetric uniform quantization converting 32-bit floating-point weights (FP32) into 8-bit signed integers (INT8):\n")
    p.add_run("   scale = max(|W|) / 127.0,   Q = clip(round(W / scale), -127, 127)\n")
    p.add_run("• Payload Packing: ").bold = True
    p.add_run("Parameters are packed as (metadata, INT8_data), where metadata stores the float scale factor and tensor shape.\n")
    p.add_run("• Compression Impact: ").bold = True
    p.add_run("Reduces payload size from 32.07 MB (FP32) down to 8.02 MB (INT8), yielding a 4.00x compression ratio (75.00% bandwidth reduction) with negligible impact on accuracy.")

    doc.add_heading("3. Formal Distinction: Privacy vs. Compression", level=3)
    p = doc.add_paragraph()
    p.add_run("• Quantization: ").bold = True
    p.add_run("Compression technique that reduces payload size and bandwidth usage. It provides no mathematical privacy guarantee; quantized weights can still leak information through inversion attacks.\n")
    p.add_run("• Differential Privacy: ").bold = True
    p.add_run("Mathematical privacy framework that bounds the influence of any single training sample on the published model outputs.")

    doc.add_heading("Likely Interview Questions (Layer B)", level=3)
    p = doc.add_paragraph()
    p.add_run("Likely L1: \"What is differential privacy?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"Differential privacy is a mathematical framework ensuring that the presence or absence of any single individual data point in a training dataset does not significantly affect the outcome of the trained model. In our system, DP-SGD achieves this by clipping per-sample gradients to a max L2 norm (C=1.0) and adding calibrated Gaussian noise (sigma) during local backpropagation. This provides a provable upper bound (epsilon, delta) on privacy leakage.\"\n")

    p.add_run("Likely L2: \"What privacy guarantee does your implementation provide, and how do you measure the privacy–accuracy trade-off?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"In our DP implementation (USE_DP=1), the platform provides (epsilon, delta)-Differential Privacy with delta = 10^-5. Over 10 federated rounds, cumulative epsilon reaches ~12.1 (or ~92.5 depending on noise multiplier sigma). We measure the privacy-accuracy trade-off by comparing our three experimental runs:\n"
              "• Pure FP32 (a_pure): Accuracy 95.2%, Payload 32.07 MB, No formal privacy guarantee (epsilon = infinity).\n"
              "• Quantized INT8 (b_quantized): Accuracy 94.8%, Payload 8.02 MB (75% reduction), No formal privacy guarantee.\n"
              "• DP-SGD + INT8 (c_dp): Accuracy 94.0%, Payload 8.02 MB, Formal (epsilon ~ 12.1, delta = 10^-5) DP guarantee.\n"
              "• Trade-off: Adding DP incurs a minor ~1.2% accuracy drop, which is acceptable given the formal mathematical privacy protection.\"")

    # ------------------ LAYER C ------------------
    doc.add_heading("Layer C — Distributed Systems and Fault Tolerance", level=2)

    doc.add_heading("1. Non-IID Data Distribution", level=3)
    p = doc.add_paragraph()
    p.add_run("In real healthcare systems, client data distributions are non-Identically and Independently Distributed (non-IID). We model this using three distinct demographic partitions (split_data.py):\n")
    p.add_run("• Hospital A: ").bold = True
    p.add_run("1,000 Normal / 250 Pneumonia (80% Normal skew)\n")
    p.add_run("• Hospital B: ").bold = True
    p.add_run("250 Normal / 1,000 Pneumonia (80% Pneumonia skew)\n")
    p.add_run("• Hospital C: ").bold = True
    p.add_run("500 Normal / 500 Pneumonia (50% Balanced)")

    doc.add_heading("2. Aggregation Strategy & Fault Tolerance (dropout_handler.py, dropout_engine.py)", level=3)
    p = doc.add_paragraph()
    p.add_run("• FedAvg: ").bold = True
    p.add_run("Weighted parameter averaging based on client sample counts (w_i = n_i / N).\n")
    p.add_run("• Client Timeout & Straggler Problem: ").bold = True
    p.add_run("Heterogeneous client hardware or network instability causes certain nodes to delay aggregation rounds.\n")
    p.add_run("• EMA-Based Completion Estimator (dropout_engine.py):\n").bold = True
    p.add_run("   - Tracks each client's historical execution time using Exponential Moving Average (mu_i) and absolute deviation (d_i):\n"
              "     mu_i = alpha * a_{i,t} + (1-alpha) * mu_i,   d_i = beta * |a_{i,t} - mu_i| + (1-beta) * d_i   (alpha=beta=0.30)\n"
              "   - Predicts finish time: T_pred,i = max(e_t, mu_i + k * d_i).\n"
              "   - If T_pred,i > hard_deadline (60s or 180s), the engine marks the client as a straggler and instructs the server to drop it preemptively without stalling the round.\n")
    p.add_run("• Quorum Protection: ").bold = True
    p.add_run("If dropping stragglers would reduce surviving clients below min_clients (Quorum K_min = ceil(0.6 * N_target) = 2), the engine overrides the drop decision and waits, prioritizing quorum over deadline.")

    doc.add_heading("Likely Interview Questions (Layer C)", level=3)
    p = doc.add_paragraph()
    p.add_run("Likely L1: \"What happens if a client disconnects?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"Our custom AdaptiveServer handles client disconnections gracefully:\n"
              "1. Gated Exception Handling: If gRPC raises UNAVAILABLE or CANCELLED, the dropout engine registers a DISCONNECTED state and increments network_failures in the client's profile.\n"
              "2. Quorum Assessment: The server checks if the remaining completed clients meet K_min (2 out of 3). If met, FedAvg aggregates the updates from surviving clients. If not, the round is aborted safely without corrupting the global model weights.\n"
              "3. Non-blocking Proxy Locks: Disconnected proxies are released so busy state does not block subsequent rounds.\"\n")

    p.add_run("Likely L2: \"How would you prevent a slow or malicious client from compromising an aggregation round?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"We apply a two-tier protection strategy:\n"
              "1. Straggler Mitigation (Slow Clients): The AdaptiveDropoutDecisionEngine continuously evaluates pending clients against T_pred = max(e_t, mu + k * d). If a client is predicted to violate the hard deadline, it is preemptively cancelled and excluded from aggregation.\n"
              "2. Poisoning Mitigation (Malicious Clients): The TrustManager computes the L2 norm of each update (||W_client - W_global||_2). If a malicious client submits poisoned weights or an unnaturally large gradient, its z-score spike relative to its own baseline drops its trust score, flagging it as UNTRUSTED for admin review.\"")

    # ------------------ LAYER D ------------------
    doc.add_heading("Layer D — Trust and Federated Unlearning", level=2)

    doc.add_heading("1. Non-IID Safe Client Trust Score (trust_manager.py)", level=3)
    p = doc.add_paragraph()
    p.add_run("Traditional anomaly detection fails in Non-IID FL because a legitimate hospital with a skewed dataset naturally produces parameter updates that differ from the global average. Our TrustManager solves this by comparing each client ONLY against its OWN past history:\n\n")
    p.add_run("Trust Score = 0.40 * S_Update + 0.30 * S_Training + 0.20 * S_History + 0.10 * S_Reliability\n\n")
    p.add_run("• Update Score (S_Update): ").bold = True
    p.add_run("Evaluates z-score of L2 update norm against the client's own historical norm mean and standard deviation.\n")
    p.add_run("• Training Score (S_Training): ").bold = True
    p.add_run("Evaluates sudden accuracy drops or loss spikes relative to the client's own historical trend.\n")
    p.add_run("• Historical Reputation (S_History): ").bold = True
    p.add_run("EMA decay (alpha = 0.30) of previous trust scores.\n")
    p.add_run("• Tagging: ").bold = True
    p.add_run(">= 80 -> TRUSTED, 50-79 -> SUSPICIOUS, < 50 -> UNTRUSTED.")

    doc.add_heading("2. Federated Unlearning & DACM Recovery (unlearn_workflow.py, dacu.py)", level=3)
    p = doc.add_paragraph()
    p.add_run("• Goal: ").bold = True
    p.add_run("Erase a target hospital's (e.g., Hospital B's) contribution post-training due to GDPR/right-to-be-forgotten requests.\n")
    p.add_run("• Step 1: Gradient Ascent Erasure (unlearn_baseline.py): ").bold = True
    p.add_run("Performs negative gradient steps (-grad L) on target client data to unlearn learned features.\n")
    p.add_run("• Step 2: Class-Distribution Audit: ").bold = True
    p.add_run("Unlearning a pneumonia-skewed client (Hospital B) leaves a severe minority-class shortage in surviving data. The server audits surviving metadata:\n"
              "   gamma = (Surviving Pneumonia Samples) / (Total Surviving Samples)\n")
    p.add_run("• Step 3: DACM Compensation Weight: ").bold = True
    p.add_run("If gamma < tau_safe (0.50), dynamic loss compensation weights are computed:\n"
              "   w_Pneumonia = 1.0 + alpha * ln(tau_safe / gamma)   (alpha=3.0)\n")
    p.add_run("• Step 4: Federated Recovery: ").bold = True
    p.add_run("Surviving clients train independent local copies with weighted CrossEntropyLoss(weight=[1.0, w_Pneumonia]), which are then aggregated via FedAvg to restore model utility without retraining from scratch.")

    doc.add_heading("Likely Interview Questions (Layer D)", level=3)
    p = doc.add_paragraph()
    p.add_run("Likely L1: \"How do you identify a suspicious client?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"We identify suspicious clients using the multi-component trust score in trust_manager.py. Rather than comparing updates across clients (which fails under Non-IID conditions), we compute a z-score of the client's L2 parameter update norm against its own historical mean (mu_norm) and standard deviation (sigma_norm). A sudden spike (z > 3.0), abrupt drop in evaluation accuracy (>20%), or sudden loss spike lowers the score below 50, automatically tagging the client as UNTRUSTED.\"\n")

    p.add_run("Likely L2: \"How do you validate that unlearning actually removes the target client's influence?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"We validate unlearning through a 3-Stage Diagnostic Audit (unlearn_workflow.py):\n"
              "1. Baseline Evaluation: Measure accuracy, recall, and F1 of the global model on target and surviving datasets before unlearning.\n"
              "2. Target Erasure Check: After gradient ascent unlearning, we evaluate the model on the target client's dataset. Target performance collapses while parameter distance (||W_unlearned - W_global||_2) confirms structural weight movement.\n"
              "3. DACM Recovery Verification: We evaluate the model on surviving clients. DACM recovery restores Recall from 0.00 (collapsed) back to 92.73%, proving that target influence is removed while surviving client utility is recovered.\"")

    # ------------------ LAYER E ------------------
    doc.add_heading("Layer E — Platform and Deployment", level=2)

    doc.add_heading("1. Web & Dashboard Architecture", level=3)
    p = doc.add_paragraph()
    p.add_run("• FastAPI Backend (dashboard_api.py / webapp/main.py): ").bold = True
    p.add_run("REST API and WebSocket connection manager (ConnectionManager) serving real-time system state (state.json) and audit logs.\n")
    p.add_run("• Flask Adapter Server (web_server.py): ").bold = True
    p.add_run("Lightweight web server hosting client gateway portals and handling local training execution and image predictions.\n")
    p.add_run("• Role-Based Access Control (RBAC):\n").bold = True
    p.add_run("   - Client Role (/hospital_a, /hospital_b, /hospital_c): Access limited strictly to Section 1: Local Training and Section 2: Upload & Verify.\n"
              "   - Server Admin Role: Full access to global aggregation commands, training parameters, and performance evaluation graphs. Client calls to /api/evaluate return HTTP 403 Forbidden.")

    doc.add_heading("2. Multi-Laptop Deployment & Testing", level=3)
    p = doc.add_paragraph()
    p.add_run("• Live Demo Wrappers (demo_server.py, demo_client.py): ").bold = True
    p.add_run("Auto-detects local LAN IPv4 addresses via socket connections, binding gRPC on 0.0.0.0:8080.\n")
    p.add_run("• Automated Experimentation Suite (scripts/run_experiments.py): ").bold = True
    p.add_run("Orchestrates automated benchmark runs (Baseline, Stragglers, Network Dropout, Recovery, Nonstationary Delays) with log parsing (parse_results.py) and paper-ready figures (generate_figures.py).\n")
    p.add_run("• Unit Test Suite (tests/): ").bold = True
    p.add_run("Standard unittest and pytest testing for the dropout decision engine, proxy locks, and quorum filtering.")

    doc.add_heading("Likely Interview Questions (Layer E)", level=3)
    p = doc.add_paragraph()
    p.add_run("Likely L1: \"How are the components connected?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"The platform connects via a 3-tier architecture:\n"
              "1. Transport Layer: Clients connect to demo_server.py over gRPC (0.0.0.0:8080) for model parameter transfer.\n"
              "2. Web Management Layer: Flask (web_server.py) on port 5000 provides RBAC client interfaces. FastAPI (dashboard_api.py) on port 8000 provides REST/WebSocket state monitoring.\n"
              "3. State Layer: dashboard_state.py provides thread-safe atomic state serialization (state.json) between FL server events and web interfaces.\"\n")

    p.add_run("Likely L2: \"How would you debug a client that cannot connect to the server on another laptop?\"\n").bold = True
    p.add_run("Answer: ").bold = True
    p.add_run("\"I would follow a structured 4-step diagnostic routine:\n"
              "1. Network Reachability: Test basic IP connectivity using 'ping <SERVER_IP>' over the shared LAN.\n"
              "2. Port Binding: Confirm the server is listening on 0.0.0.0:8080 rather than 127.0.0.1 using 'netstat -ano | findstr 8080' or 'Test-NetConnection -ComputerName <SERVER_IP> -Port 8080'.\n"
              "3. Firewall Rules: Check if Windows Defender Firewall is blocking inbound TCP traffic on port 8080 on the server laptop.\n"
              "4. gRPC Configuration: Inspect client console logs for gRPC status codes (UNAVAILABLE vs DEADLINE_EXCEEDED) and verify that SERVER_ADDRESS matches <SERVER_IP>:8080.\"")

    # -------------------------------------------------------------
    # PART 2: THE 10 CRUCIAL QUESTIONS
    # -------------------------------------------------------------
    h1_p2 = doc.add_heading("PART 2: THE 10 CRUCIAL QUESTIONS", level=1)
    h1_p2.paragraph_format.space_before = Pt(20)
    h1_p2.paragraph_format.space_after = Pt(12)

    q1 = "What do you mean by 'decentralized' in this implementation?"
    a1 = ("In our implementation, 'decentralized' refers to decentralized data storage and local model execution. "
          "Patient images remain strictly within the isolated environment of each hospital node. However, the aggregation "
          "topology is federated with a central server coordinator (demo_server.py), which coordinates rounds and executes "
          "FedAvg. It is not a peer-to-peer (P2P) blockchain topology.")
    add_qa_box(1, q1, a1)

    q2 = "Why are hospitals A, B, and C given different class distributions?"
    a2 = ("They are given different class distributions to mirror real-world non-IID clinical demographic skew:\n"
          "• Hospital A is a general screening facility (80% Normal / 20% Pneumonia).\n"
          "• Hospital B is a specialized pulmonology referral center (20% Normal / 80% Pneumonia).\n"
          "• Hospital C is a balanced regional clinic (50% Normal / 50% Pneumonia).\n"
          "Testing on non-IID distributions verifies that our global FL model generalizes across diverse populations, "
          "and that our trust module correctly avoids penalizing legitimate demographic skew.")
    add_qa_box(2, q2, a2)

    q3 = "Does federated learning automatically guarantee privacy?"
    a3 = ("No. FL prevents raw data exposure by keeping images on local devices, but parameter updates or gradients "
          "transmitted to the server can still leak private information through model inversion or gradient leakage attacks. "
          "Formal privacy guarantees require adding Differential Privacy (DP-SGD), which clips gradients and injects noise.")
    add_qa_box(3, q3, a3)

    q4 = "Does INT8 quantization itself protect sensitive data?"
    a4 = ("No. INT8 quantization is strictly a compression and bandwidth reduction technique that maps FP32 weights to "
          "8-bit integers (quantization.py), reducing model size by 75%. It does not obscure weight relationships mathematically "
          "and provides no formal privacy protection.")
    add_qa_box(4, q4, a4)

    q5 = "How is the global model updated after local training?"
    a5 = ("1. In fit(), clients train ChestCNN locally on their dataset for 2 epochs.\n"
          "2. Clients convert weights to NumPy arrays, apply INT8 quantization, and transmit the parameters and sample count (n_i) to the server via gRPC.\n"
          "3. The server dequantizes parameters back to FP32.\n"
          "4. SaveModelStrategy executes FedAvg aggregation: W_global = sum( (n_i / N_total) * W_i ).\n"
          "5. The aggregated parameters are loaded into the global ChestCNN model and saved to disk.")
    add_qa_box(5, q5, a5)

    q6 = "Why could non-IID data make legitimate clients appear suspicious?"
    a6 = ("In traditional anomaly detection, a client's update is compared against the average update of all clients. "
          "Under non-IID conditions, a hospital with a unique demographic skew (e.g., Hospital B's 80% pneumonia rate) "
          "naturally produces weight updates that deviate significantly from the global mean. Comparing clients to the group "
          "would falsely flag Hospital B as malicious or 'suspicious'. Our implementation avoids this by evaluating each "
          "client only against its own historical baseline (trust_manager.py).")
    add_qa_box(6, q6, a6)

    q7 = "What happens if too few clients survive an aggregation round?"
    a7 = ("If the number of successful client updates falls below min_clients (K_min = 2), AdaptiveServer.fit_round() detects "
          "a quorum failure, prints an error log, and aborts aggregation for that round. The global model parameters remain "
          "unchanged from the previous round, preventing corrupted or unrepresentative updates from altering the model.")
    add_qa_box(7, q7, a7)

    q8 = "What limitations does gradient-ascent unlearning have?"
    a8 = ("Gradient ascent (-grad L) pushes model parameters aggressively away from the target client's dataset. Its main limitations are:\n"
          "1. Catastrophic Forgetting / Model Collapse: Pushing gradients in reverse can destroy useful representations learned for other clients, causing overall recall to drop to 0.00.\n"
          "2. Sensitivity to Hyperparameters: Requires precise tuning of learning rate and ascent steps (UNLEARN_LR, UNLEARN_STEPS).\n"
          "3. Lack of Exact Mathematical Erasure: It is an approximate erasure technique, unlike retraining from scratch.")
    add_qa_box(8, q8, a8)

    q9 = "How do you measure model utility before and after unlearning?"
    a9 = ("We measure utility across 3 stages using evaluate_unlearning.py and unlearn_workflow.py:\n"
          "1. Stage 1 (Before Unlearning): Measure Accuracy, Precision, Recall, and F1 of the converged global model across target and surviving clients.\n"
          "2. Stage 2 (Post Gradient Ascent): Re-evaluate to confirm that target client performance drops while tracking L2 parameter distance (||W_unlearned - W_global||_2) to detect potential model collapse.\n"
          "3. Stage 3 (Post DACM Recovery): Evaluate the recovered model on surviving clients to verify that Recall and F1 return to baseline levels (>= 92% Recall, >= 87% F1).")
    add_qa_box(9, q9, a9)

    q10 = "What did you personally implement, test and verify?"
    a10 = ("• Implemented: The complete end-to-end Python codebase, including ChestCNN model training (model.py), INT8 quantization routines (quantization.py), Opacus DP-SGD integration, the non-IID data generator (split_data.py), the EMA-based Adaptive Dropout Engine (dropout_engine.py), the non-IID safe Trust Scoring module (trust_manager.py), the post-unlearning DACM recovery protocol (dacu.py, unlearn_workflow.py), and the Flask/FastAPI RBAC web dashboards.\n"
           "• Tested & Verified:\n"
           "   - Verified DP-SGD privacy accounting and INT8 payload compression (32.07 MB -> 8.02 MB).\n"
           "   - Executed 8 automated experiment scenarios (scripts/run_experiments.py) validating straggler handling and network reconnection.\n"
           "   - Ran unit test suites (test_dropout_engine.py, test_dropout_handler.py) covering quorum protection and proxy locks.\n"
           "   - Verified global model accuracy (94.00%), F1 (87.18%), and ROC-AUC (0.9803) on held-out test sets (evaluate.py, evaluate_comparison.py).")
    add_qa_box(10, q10, a10)

    # Save document as .docx and .doc
    docx_path = Path("D:/AKASH_SAKEC/AK_Project/Z/Federated-Learning-main/Federated_Healthcare_AI_Architectural_Breakdown.docx")
    doc_path = Path("D:/AKASH_SAKEC/AK_Project/Z/Federated-Learning-main/Federated_Healthcare_AI_Architectural_Breakdown.doc")

    doc.save(str(docx_path))
    doc.save(str(doc_path))
    print(f"Successfully generated Word documents:\n1. {docx_path}\n2. {doc_path}")

if __name__ == "__main__":
    create_document()
