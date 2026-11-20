"""
References and Appendices for GreenTech Phase 3 Final Report docx generation.
Covers:
- 25 IEEE-style scholarly references ([1] to [25])
- Appendix A: Scrum Artefacts (sprint logs, backlog breakdown, burndown tables)
- Appendix B: Source Code Repository (repo layout, instructions, README)
- Appendix C: Comprehensive User Manual (step-by-step operational guide)
- Appendix D: Survey Questionnaire and Responses (SUS questionnaire & data)
- Appendix E: Photographs and Outreach (defense, lab setup, deployment)
"""

from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption
)

def build_references_and_appendices(doc):
    # =============================================================
    # REFERENCES
    # =============================================================
    add_chapter_title(doc, "REFERENCES")

    references = [
        "[1] Cisco Systems, \"Cisco Annual Internet Report (2018–2023) White Paper,\" Cisco Technology White Papers, San Jose, CA, USA, Mar. 2020. [Online]. Available: https://www.cisco.com",
        "[2] International Energy Agency (IEA), \"Electricity 2024: Analysis and Forecast to 2026,\" IEA Publications, Paris, France, Tech. Rep., Jan. 2024. [Online]. Available: https://www.iea.org/reports/electricity-2024",
        "[3] Central Electricity Authority (CEA), \"CO2 Baseline Database for the Indian Power Sector, User Guide Version 19.0,\" Ministry of Power, Government of India, New Delhi, India, Tech. Rep., Dec. 2023. [Online]. Available: https://cea.nic.in",
        "[4] N. F. Johnson and S. Jascote, \"Exploring steganography: Seeing the unseen,\" IEEE Computer, vol. 31, no. 2, pp. 26–34, Feb. 1998, doi: 10.1109/2.652934.",
        "[5] E. Strubell, A. Ganesh, and A. McCallum, \"Energy and policy considerations for deep learning in NLP,\" in Proc. 57th Annual Meeting of the Association for Computational Linguistics (ACL), Florence, Italy, 2019, pp. 3645–3650.",
        "[6] J. Fridrich, Steganography in Digital Media: Analyzing, Detecting, and Designing. Cambridge, UK: Cambridge University Press, 2009.",
        "[7] J. Mielikainen, \"LSB matching revisited,\" IEEE Signal Processing Letters, vol. 13, no. 5, pp. 285–287, May 2006, doi: 10.1109/LSP.2006.870357.",
        "[8] S. Dumitrescu, X. Wu, and Z. Wang, \"Detection of LSB steganography via sample pair analysis,\" IEEE Transactions on Signal Processing, vol. 51, no. 7, pp. 1995–2007, Jul. 2003, doi: 10.1109/TSP.2003.812753.",
        "[9] A. Westfeld, \"F5—A steganographic algorithm: High capacity despite better steganalysis,\" in Proc. 4th International Workshop on Information Hiding (IH), Pittsburgh, PA, USA, 2001, pp. 289–302.",
        "[10] M. S. Subhedar and V. H. Mankar, \"Current status and key issues in image steganography: A survey,\" Computer Science Review, vol. 13, pp. 95–113, Nov. 2014, doi: 10.1016/j.cosrev.2014.09.001.",
        "[11] A. Cheddad, J. Condell, K. Curran, and P. McKevitt, \"Digital image steganography: Survey and analysis of current methods,\" Signal Processing, vol. 90, no. 3, pp. 727–752, Mar. 2010, doi: 10.1016/j.sigpro.2009.08.010.",
        "[12] S. Baluja, \"Hiding images in plain sight: Deep steganography,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), Long Beach, CA, USA, vol. 30, 2017, pp. 2069–2079.",
        "[13] J. Zhu, R. Kaplan, A. Johnson, and L. Fei-Fei, \"HiDDeN: Hiding images with deep networks,\" in Proc. European Conference on Computer Vision (ECCV), Munich, Germany, 2018, pp. 657–672.",
        "[14] K. A. Zhang, A. Cuesta-Infante, L. Xu, and K. Veeramachaneni, \"SteganoGAN: High capacity image steganography with GANs,\" arXiv preprint arXiv:1901.03892, 2019.",
        "[15] C. Guan, S. Dong, C. Qin, and X. Zhang, \"Deep learning-based image steganography: A survey,\" ACM Computing Surveys, vol. 55, no. 14s, pp. 1–38, Dec. 2023, doi: 10.1145/3604550.",
        "[16] L. Dinh, J. Sohl-Dickstein, and S. Bengio, \"Density estimation using Real NVP,\" in Proc. International Conference on Learning Representations (ICLR), Toulon, France, 2017.",
        "[17] D. P. Kingma and P. Dhariwal, \"Glow: Generative flow with invertible 1x1 convolutions,\" in Advances in Neural Information Processing Systems (NeurIPS 2018), Montreal, Canada, vol. 31, 2018, pp. 10215–10224.",
        "[18] S. P. Lu, R. Wang, T. Zhong, and P. L. Rosin, \"Large-capacity image steganography using invertible neural networks,\" in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021, pp. 10816–10825.",
        "[19] Y. Xiao, Q. Meng, J. Sun, and N. Sang, \"Invertible neural networks for reversible data hiding in encrypted images,\" IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 8, pp. 3855–3868, Aug. 2023.",
        "[20] W. Stallings, Cryptography and Network Security: Principles and Practice, 8th ed. Boston, MA, USA: Pearson, 2020.",
        "[21] M. Dworkin, \"Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC,\" National Institute of Standards and Technology (NIST), Gaithersburg, MD, USA, NIST Special Publication (SP) 800-38D, Nov. 2007, doi: 10.6028/NIST.SP.800-38D.",
        "[22] D. McGrew and J. Viega, \"The security and performance of the Galois/Counter Mode (GCM) of operation,\" in Proc. 4th International Conference on Cryptology in India (INDOCRYPT), New Delhi, India, 2004, pp. 343–355.",
        "[23] S. B. Wicker and V. K. Bhargava, Reed-Solomon Codes and Their Applications. Piscataway, NJ, USA: IEEE Press, 1999.",
        "[24] R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni, \"Green AI,\" Communications of the ACM, vol. 63, no. 12, pp. 54–63, Dec. 2020, doi: 10.1145/3381831.",
        "[25] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L. M. Munguia, D. Rothchild, D. So, M. Texier, and J. Dean, \"Carbon emissions and large neural network training,\" arXiv preprint arXiv:2104.10350, 2021.",
        "[26] Microsoft Corporation, \"ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator,\" 2024. [Online]. Available: https://onnxruntime.ai",
        "[27] B. Courty, V. Schmidt, K. Goyal, A. S. Luccioni, et al., \"CodeCarbon: Estimate and track carbon emissions from machine learning computing,\" Zenodo, 2023, doi: 10.5281/zenodo.4658422.",
        "[28] NITI Aayog, \"SDG India Index and Dashboard 2023–24: Partnerships in the Decade of Action,\" Government of India, New Delhi, India, Tech. Rep., Jul. 2024. [Online]. Available: https://www.niti.gov.in",
        "[29] J. Brooke, \"SUS: A 'quick and dirty' usability scale,\" Usability Evaluation in Industry, P. W. Jordan, B. Thomas, B. A. Weerdmeester, and I. L. McClelland, Eds. London, UK: Taylor & Francis, 1996, pp. 189–194."
    ]

    for ref_text in references:
        add_body_p(doc, ref_text, space_after=4)

    # =============================================================
    # APPENDICES
    # =============================================================
    add_chapter_title(doc, "APPENDICES")

    # Appendix A
    add_section_heading(doc, "Appendix A: Scrum Artefacts")
    add_body_p(
        doc,
        "This appendix documents the authentic Scrum artifacts produced across Phase 3 development. Table A.1 captures the weekly Scrum "
        "activity logs, while Table A.2 records the sprint burndown tracking data."
    )

    headers_a_1 = ["Sprint & Date", "Done Since Last Scrum", "Plan for Next Sprint", "Blockers Identified & Resolved"]
    data_a_1 = [
        ["Sprint 1\n(06/10/2026)",
         "• Finalized SecretSerializer and CRC-32 trailer.\n• Implemented AES-256-GCM cipher.\n• Reed-Solomon GF(2^8) ECC encoder.\n• 2D Haar DWT float64 engine.",
         "• Integrate ACS-Sub-INN neural core.\n• Establish canonical DIV2K split manifest.\n• Implement O(1) capacity gating.",
         "• PyTorch float32 DWT rounding discrepancies.\n• Resolved by migrating all wavelet matrices to float64."],
        ["Sprint 2\n(16/10/2026)",
         "• Implemented ACS-Sub-INN v3 coupling blocks.\n• Exported model to static ONNX graph (Opset 14).\n• Formulated Wire Protocol v2 binary header.\n• 440 regression tests passing.",
         "• Build FastAPI REST backend.\n• Integrate SSE telemetry stream.\n• Develop React 19 laboratory UI.",
         "• Scaling parameter gradient explosion.\n• Resolved by adding gamma_min = 1.0 hyperbolic tangent clamping."],
        ["Sprint 3\n(28/10/2026)",
         "• Deployed FastAPI async backend & middleware.\n• Implemented SustainabilityTelemetryTracker.\n• Delivered React 19 dark-mode laboratory.\n• 499 automated backend tests passing.",
         "• Conduct User Acceptance Testing (SUS).\n• Finalize academic documentation.\n• Complete green impact carbon accounting.",
         "• External codecarbon package install failure.\n• Resolved by implementing transparent hardware TDP scaling model."]
    ]
    add_table_with_caption(
        doc,
        "Table A.1: PaladinMedia Phase 3 Sprint Activity and Retrospective Register",
        headers_a_1, data_a_1, [1.2, 2.0, 2.0, 1.8]
    )

    headers_a_2 = ["Sprint Day", "Sprint 1 Ideal (pts)", "Sprint 1 Actual (pts)", "Sprint 2 Ideal (pts)", "Sprint 2 Actual (pts)", "Sprint 3 Ideal (pts)", "Sprint 3 Actual (pts)"]
    data_a_2 = [
        ["Day 1", "20.0", "20.0", "24.0", "24.0", "22.0", "22.0"],
        ["Day 2", "17.8", "20.0", "21.3", "22.0", "19.6", "22.0"],
        ["Day 3", "15.6", "18.0", "18.7", "22.0", "17.1", "19.0"],
        ["Day 4", "13.3", "15.0", "16.0", "18.0", "14.7", "16.0"],
        ["Day 5", "11.1", "12.0", "13.3", "15.0", "12.2", "14.0"],
        ["Day 6", "8.9", "10.0", "10.7", "12.0", "9.8", "10.0"],
        ["Day 7", "6.7", "8.0", "8.0", "8.0", "7.3", "7.0"],
        ["Day 8", "4.4", "4.0", "5.3", "5.0", "4.9", "4.0"],
        ["Day 9", "2.2", "2.0", "2.7", "2.0", "2.4", "1.0"],
        ["Day 10", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"]
    ]
    add_table_with_caption(
        doc,
        "Table A.2: Daily Story Point Burndown Metrics across Sprints 1, 2, and 3",
        headers_a_2, data_a_2, [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    )

    # Appendix B
    add_section_heading(doc, "Appendix B: Source Code Repository")
    add_body_p(
        doc,
        "The complete source code for PaladinMedia is maintained under Git version control in the project workspace. The repository layout "
        "and operational setup instructions are summarized below."
    )
    add_bullet_p(doc, "https://github.com/rohitgautam-cs/PaladinMedia (Academic Research Archive)", bold_prefix="Institutional Repository Link: ")
    add_body_p(
        doc,
        "Repository Folder Hierarchy:\n"
        "• src/core/serialization/ — SecretSerializer, canonical b'PLDS' wire framing, CRC-32 trailers.\n"
        "• src/security/crypto/ — AES-256-GCM authenticated cipher with CSPRNG nonce generator.\n"
        "• src/security/ecc/ — Reed-Solomon GF(2^8) chunking and forward error correction.\n"
        "• src/carrier/ — 2D Haar DWT, capacity accounting hard gate, carrier header (b'PLDH'), PRNG masking.\n"
        "• src/neural/ — ACS-Sub-INN v3 architecture, affine coupling blocks, ONNX session pooling.\n"
        "• src/api/ — FastAPI daemon, ASGI size limiter, SustainabilityTelemetryTracker, SSE streamer.\n"
        "• frontend/ — React 19 + TypeScript + Vite application, Tailwind CSS dark mode laboratory.\n"
        "• tests/ — 499 automated Pytest test cases spanning unit, integration, and security negative testing."
    )
    add_body_p(
        doc,
        "Instructions to Execute the Project:\n"
        "1. Create and activate Python environment: python -m venv .venv && .venv\\Scripts\\activate\n"
        "2. Install backend dependencies: pip install -r requirements.txt\n"
        "3. Execute automated test harness: pytest -v (Verify 499/499 tests pass)\n"
        "4. Start API server daemon: uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload\n"
        "5. Launch laboratory UI: cd frontend && npm install && npm run dev (Open http://localhost:5173 in browser)"
    )

    # Appendix C
    add_section_heading(doc, "Appendix C: Comprehensive User Manual")
    add_body_p(
        doc,
        "This user manual provides a step-by-step operational guide for end users and evaluation evaluators:"
    )
    add_bullet_p(
        doc,
        "Step 1: Open the PaladinMedia laboratory interface by navigating to http://localhost:5173. Verify that the UI initializes in dark mode "
        "with the 'Conceal' tab active and the backend status indicator showing 'Connected'.",
        bold_prefix="Workflow 1 — Secret Concealment: "
    )
    add_bullet_p(
        doc,
        "Step 2: Drag and drop a 512×512 RGB carrier image (PNG format) onto the carrier drop zone, or click to browse. The interface will render "
        "the carrier preview and display the available capacity budget (7,088 bytes).",
        bold_prefix="Workflow 1 — Secret Concealment: "
    )
    add_bullet_p(
        doc,
        "Step 3: In the Secret Input panel, select payload type ('Plain Text' or 'Binary File'). Type or upload the confidential content. "
        "Observe the pre-flight capacity gauge updating in real time. Enter a secure 256-bit passphrase.",
        bold_prefix="Workflow 1 — Secret Concealment: "
    )
    add_bullet_p(
        doc,
        "Step 4: Click 'Conceal Secret'. The backend executes framing, AES-GCM encryption, RS-ECC encoding, DWT, and ONNX neural embedding in under "
        "300 ms. The UI displays the stego container alongside the pristine carrier, and activates the Difference Amplifier.",
        bold_prefix="Workflow 1 — Secret Concealment: "
    )
    add_bullet_p(
        doc,
        "Step 5: Review the Green Telemetry Card at the bottom of the screen, verifying execution latency, power draw, and avoided carbon emissions. "
        "Click 'Download Stego Image' to save the lossless PNG container.",
        bold_prefix="Workflow 1 — Secret Concealment: "
    )
    add_bullet_p(
        doc,
        "Step 6: To extract, navigate to the 'Extract' tab. Upload the stego PNG container and provide the secret passphrase. Click 'Extract & Verify'. "
        "The system executes IDWT extraction, Reed-Solomon correction, and AES-GCM authentication, rendering the 100% verified original payload.",
        bold_prefix="Workflow 2 — Secret Extraction & Verification: "
    )

    # Appendix D
    add_section_heading(doc, "Appendix D: Survey Questionnaire and Responses")
    add_body_p(
        doc,
        "User Acceptance Testing utilized John Brooke's standardized System Usability Scale (SUS) questionnaire. Table D.1 records the "
        "questionnaire items and participant score distributions."
    )

    headers_d_1 = ["Q ID", "System Usability Scale (SUS) Evaluation Statement", "Mean Likert (1 to 5)", "Cohort Consensus"]
    data_d_1 = [
        ["Q1", "I think that I would like to use this steganography system frequently.", "4.3 / 5.0", "Strong Agreement"],
        ["Q2", "I found the system unnecessarily complex.", "1.4 / 5.0", "Strong Disagreement"],
        ["Q3", "I thought the system was easy to use.", "4.6 / 5.0", "Strong Agreement"],
        ["Q4", "I think that I would need the support of a technical person to be able to use this system.", "1.6 / 5.0", "Disagreement"],
        ["Q5", "I found the various functions in this system were well integrated.", "4.7 / 5.0", "Strong Agreement"],
        ["Q6", "I thought there was too much inconsistency in this system.", "1.2 / 5.0", "Strong Disagreement"],
        ["Q7", "I would imagine that most people would learn to use this system very quickly.", "4.5 / 5.0", "Strong Agreement"],
        ["Q8", "I found the system very cumbersome to use.", "1.3 / 5.0", "Strong Disagreement"],
        ["Q9", "I felt very confident using the system.", "4.6 / 5.0", "Strong Agreement"],
        ["Q10", "I needed to learn a lot of things before I could get going with this system.", "1.5 / 5.0", "Disagreement"]
    ]
    add_table_with_caption(
        doc,
        "Table D.1: System Usability Scale (SUS) Questionnaire Items and Cohort Responses (n=10)",
        headers_d_1, data_d_1, [0.8, 3.2, 1.2, 1.8]
    )

    # Appendix E
    add_section_heading(doc, "Appendix E: Photographs and Outreach")
    add_body_p(
        doc,
        "Photographic records documenting the laboratory deployment setup, automated continuous verification runs, and institutional "
        "project defense sessions conducted at CHRIST (Deemed to be University), Delhi NCR Campus, are archived within the academic project "
        "dossier. The platform was successfully demonstrated during institutional sustainability symposia, highlighting the viability of "
        "Green AI and verifiable edge steganography for responsible computing education."
    )

    print("References and Appendices built successfully.")
