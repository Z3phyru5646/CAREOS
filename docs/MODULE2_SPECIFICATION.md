Architectural Blueprint and Feature Specification for CARE'OS Module 2: Ambient AI Scribe with 2nd-Pass Clinical Verifier
To transform Module 2 (AI Scribe with 2nd-Pass Verifier) of your CARE'OS project (CSET343 – AI In Healthcare) from a basic speech-to-text summarizer into a publication-grade clinical engineering system, you must architect a five-stage, ontology-grounded pipeline that separates clinical abstraction from factual verification. Standard ambient scribes pipe raw Automatic Speech Recognition (ASR) transcripts directly into a Large Language Model (LLM) to draft a Subjective, Objective, Assessment, and Plan (SOAP) note. This naive approach fails in clinical practice for two reasons: according to clinical speech engineering analyses from Corti and AssemblyAI, general ASR models can exhibit word error rates (WER) of 40% or more on specialized medical dictation and sound-alike pharmaceuticals (such as substituting the mealtime insulin Humalog for the immunosuppressant Humira), while standard literal fact-checking frameworks falsely penalize valid clinical reasoning—yielding an inflated 35.2% apparent hallucination rate because they cannot distinguish between a dangerous fabrication and a legitimate medical translation (such as mapping "pain after I eat" to "postprandial pain").   

Your implementation of Module 2 resolves both failure modes by pairing a Speaker-Attributed Medical ASR and Clinical Named Entity Recognition (NER) front-end with a Dual-Tier, Inference-Aware 2nd-Pass Verifier. In the first pass, WhisperX (using faster-whisper large-v3 and wav2vec2.0 forced phoneme alignment) and pyannote.audio (speaker-diarization-community-1 or DiariZen) generate word-stamped, speaker-attributed transcripts. A deterministic clinical NLP layer (Med7, scispaCy, and negspaCy) extracts affirmed and negated symptoms, posology attributes, and Unified Medical Language System (UMLS) / RxNorm concept identifiers (CUIs) to ground an open-weight medical LLM (MedGemma 1.5 or OpenBioLLM) via Pydantic and Instructor constrained JSON decoding. In the second pass, the verifier decomposes the draft SOAP note into atomic claims and routes them through a Fast-Path Natural Language Inference (NLI) cross-encoder (MiniCheck-DeBERTa-v3-Large and deberta-v3-mednli-scifact-open-sentence-nli) followed by a Slow-Path Ontology-Grounded Retrieval-Augmented Generation (RAG) Judge. Grounded in SNOMED CT, ICD-10, and RxNorm, this two-stage verification methodology reduces false-positive hallucination flags from 35.2% down to 9.1%—closely matching human physician evaluators (10.4%) while retaining complete agreement on true medication, dosage, and diagnostic fabrications.   

To make your CSET343 project stand out during academic evaluation and live demonstration, this blueprint equips Module 2 with five high-impact innovations: (1) a Code-Switched "Hinglish" Clinical ASR Engine (Trelis/whisper-hinglish-preview and EEND-VC) tailored for Indian outpatient clinics where doctors and patients mix Hindi and English mid-sentence; (2) an interactive "Linked Evidence" Audio-Timestamp Provenance UI that lets clinicians click any SOAP sentence to replay the exact audio segment; (3) a live "Clinical Co-Pilot" Whisperer that surfaces missing red-flag questions and differential diagnoses during the consultation; (4) a Bidirectional Inter-Module Data Bus that automatically triggers Module 3 (Genomics-Informed Treatment Recommendation) when any of 30+ Clinical Pharmacogenetics Implementation Consortium (CPIC) Level A drugs are prescribed, while streaming live encounter complexity metrics back to Module 1 (Queue and Wait-Time Predictor); and (5) a Dual-Artifact Post-Visit Exporter that generates both HL7 FHIR R4 structured bundles and plain-language patient education summaries.   

End-to-End System Architecture of Module 2
A professional ambient clinical intelligence system cannot operate as a monolithic prompt wrapper; it requires modular decoupling so each component can be independently benchmarked, audited, and upgraded. Your Module 2 architecture is structured into five sequential processing stages connected to a real-time inter-module event bus.   


Core Functional Pipeline vs. Advanced Project Extensions
To help you scope your development milestones across the semester, Table 1 contrasts the baseline requirements typically expected in an undergraduate healthcare AI project against the production-grade engineering stack and novel extensions detailed in this report.

Subsystem Layer	Basic Student Implementation	Professional CARE'OS Module 2 Stack	Standout Novelty for CSET343 Defense
1. Audio & Diarization	
Single-speaker microphone input passed to base OpenAI Whisper API; wall-of-text transcript.

16 kHz resampling, EQ filtering, WhisperX (faster-whisper large-v3 + wav2vec2.0 forced phoneme alignment) + pyannote community-1 / DiariZen.

Hinglish Code-Switched Pipeline: Trelis/whisper-hinglish-preview with `<
2. Speaker Role & ASR Correction	
Manual labels (Speaker 1, Speaker 2); raw ASR spelling mistakes left untouched.

Sliding-window Speaker Role Identification (Doctor, Patient, Medical Assistant, Other) + CharBERTDrug / LLM phonetic correction against 69,824 RxNorm drug strings.

Real-Time Clinical Co-Pilot: Live consultation listener that prompts the doctor with unasked red-flag questions and differentials.

3. Clinical Entity Pre-Extraction	None; relies entirely on the LLM prompt to spot medications and symptoms.	
Deterministic spaCy pipeline: Med7 (7 posology tags), scispaCy (AbbreviationDetector + UmlsEntityLinker), and negspaCy (NegEx clinical negation).

Pertinent Negative Protection: Explicit pre-tagging of denied symptoms (negex=True) prevents the LLM from flipping negated findings into active diagnoses.

4. SOAP Note Generation	
Unconstrained free-text ChatGPT prompt returning markdown paragraphs.

Section-by-section generation using MedGemma 1.5 (4B/27B) or OpenBioLLM with Instructor + Pydantic constrained JSON schema enforcement.

Linked Evidence Provenance: Every generated SOAP sentence embeds exact word-level audio start/end timestamps for one-click audio replay.

5. 2nd-Pass Verifier	Simple second LLM prompt asking "Are there any hallucinations in this note?"	
Dual-Tier Verifier: Atomic claim splitter → Fast-Path MiniCheck-DeBERTa-v3 / MedNLI → Slow-Path Ontology-RAG Judge (5-Tier Taxonomy + MEDEC check).

Cross-Module CARE'OS Bus: Live Module 3 CPIC Level A pharmacogenomic alert injection + Module 1 live queue wait-time telemetry.

  
Stage 1 and Stage 2: Clinical Audio Ingestion, Speaker Role Diarization, and Medical ASR Post-Correction
Acoustic Pre-Processing and Overlap-Aware Speaker Diarization
Clinical consultations are acoustically hostile environments characterized by far-field microphone placement, ambient hospital noise, and rapid, uneven turn-taking between stressed patients and clinicians. Before feeding audio into neural models, your ingestion pipeline must standardize raw streams into single-channel (mono) WAV audio sampled at 16 kHz, automatically downmixing stereo or multi-channel recordings by averaging channels. Implementing a lightweight digital signal processing (DSP) pre-filter—specifically high-pass and low-pass filtering paired with gentle equalization attenuation in the 300–400 Hz range to remove acoustic muddiness, a 4 kHz boost to sharpen consonant intelligibility, and peak normalization smoothing—significantly stabilizes downstream Voice Activity Detection (VAD).   

Once pre-processed, the audio enters the diarization and transcription engine. Standard OpenAI Whisper outputs utterance-level timestamps that can drift by several seconds and lacks native speaker diarization, producing an undifferentiated wall of text. You should implement the WhisperX two-pass architecture:   

VAD-Cut Batched ASR (faster-whisper): Rather than using standard buffered transcription, slice the audio along neural VAD boundaries and run batched single-pass inference (--without_timestamps True) using faster-whisper (large-v3 in float16 precision), which achieves up to a 70x speed-up while lowering Word Error Rate (WER). Set --condition_on_prev_text False by default to reduce hallucination loops on noisy audio, or enable WhisperX's 2026 interleaved_context=True (condition_on_previous_context) option during clean technical dictation to improve punctuation and proper noun adherence.   

Phoneme-Level Forced Alignment (wav2vec2.0): Pass the orthographic transcript and audio segments into a fine-tuned phoneme-based ASR model (wav2vec2.0) to execute dynamic programming forced alignment. Installing numba (pip install "whisperx[fast]") compiles the alignment loops Just-In-Time (JIT), accelerating word-level timestamp alignment by 30x to 100x.   

Neural Speaker Diarization (pyannote Community-1 / DiariZen): In parallel, partition the audio stream into homogeneous speaker segments via a four-stage neural diarization pipeline: neural VAD, speaker change/turn segmentation, high-dimensional vocal fingerprint embedding extraction, and clustering. Rather than legacy pyannote/speaker-diarization-3.1, integrate pyannote/speaker-diarization-community-1, which features improved speaker counting, native offline execution, and exclusive speaker diarization mode designed specifically to simplify timestamp reconciliation with transcription words. Across academic benchmarks evaluated with no forgiveness collar and full overlap scoring (updated September 2025), community-1 reduces Diarization Error Rate (DER) over legacy 3.1 from 18.8% to 17.0% on AMI (IHM), from 24.5% to 20.3% on AliMeeting, and from 21.4% to 20.2% on DIHARD 3 (while the cloud-hosted precision-2 model reaches 12.9%, 15.2%, and 14.7%, respectively).   

Overlapping Speech Resolution: In real primary-care consultations, speaker overlap accounts for approximately 4.35% of total session duration (alongside 84.30% single-speaker speech and 11.35% silence). When a doctor interrupts a patient or a family member speaks simultaneously, simple temporal intersection matching struggles. You can handle overlaps either by integrating a Conv-TasNet source separation module with dynamic speaker profile updating (creating a new speaker profile when cosine similarity falls below a threshold of 0.7), or by adopting an End-to-End Neural Diarization with Vector Clustering (EEND-VC) model such as DiariZen (combining WavLM acoustic features with a Conformer network and k-means clustering set to k = 2 for doctor and patient). On the 2026 DISPLACE-M clinical dialogue benchmark, domain-adapted DiariZen achieved a DER between 7.81% and 8.57% (reaching 8.48% when fused via DOVER-Lap).   

Automated Speaker Role Identification (SRI)
Diarization assigns anonymous cluster IDs (SPEAKER_00, SPEAKER_01) rather than clinical roles. Attributing a patient's subjective complaint to the physician—or a physician's casual hypothesis to a confirmed diagnosis—corrupts the SOAP note. In outpatient clinics, conversations routinely involve up to four distinct roles: doctor, patient, medical assistant, and other (such as a spouse or caregiver speaking in the room).   

To automate Speaker Role Identification (SRI), implement a sliding context-window classifier over the diarized utterances. For each target utterance, construct a causal context window consisting only of the preceding utterances within a fixed window size (plus the target utterance at the end of the window) so that inputs fit strictly within transformer context limits while keeping the target utterance unambiguous. By combining conversational turn-taking cues (such as tone, sentiment, diction, and clinical terminology density) with an LLM or fine-tuned sequence classifier, the pipeline maps anonymous speaker clusters to Doctor, Patient, Medical Assistant, or Other. In empirical evaluations of zero-shot LLM diarization and correction across clinical transcripts, pairing Whisper 1 with Gemini Pro achieved a permutation-based Word Error Rate (P-WER) of 17.90% ± 6.83%, outperforming standalone Deepgram Nova 2 (18.02% ± 6.55%) and Soniox (18.08% ± 13.77%).   

Medical ASR Post-Error Correction and RxNorm Phonetic Grounding
General-purpose ASR models optimize for overall Word Error Rate, which masks catastrophic failures on low-frequency medical terminology, anatomical terms, and pharmaceutical names. According to a September 2026 vendor benchmark by Omi Health Research across 1,513 sealed clinical audio clips, even when leading speech recognition systems cluster near ~6% overall WER, their medical-term WER (M-WER), drug-name error rate, and dosage-event F1 diverge substantially. Furthermore, an AssemblyAI analysis of pharmaceutical speech recognition (updated June 2026) categorizes clinical ASR drug errors into four distinct clinical risk classes:   

Real-Drug Substitution Errors (Critical Clinical Risk / Often Passes Review): When encountering an unfamiliar drug name, the ASR model forces the acoustic signal into a phonetically similar real medication from a different therapeutic class. Documented examples include transcribing "Humira" (an immunosuppressant biologic) as "Humalog" (a diabetes insulin medication), "Zantac" (an antacid/heartburn drug) as "Xanax" (an anti-anxiety medication), or "ofatumumab" as "ocrelizumab" (two anti-CD20 biologics with different dosing schedules and indications). Because the substituted word is a real medication, spell-checkers and rapid manual reviews frequently miss it.   

Phonetic Spelling Errors (Moderate Risk): Recognizable misspellings that break downstream automated EHR parsing or Module 3 genomic triggers—such as transcribing "metformin" as "metforman", or Long-Term Care Data Cooperative (LTCDC) clinical repository errors such as "simethacone" for "simethicone", "morphin sulfate" for "morphinesulfate", "Zorfan" for "Zofran", and "levison" for "Levsin".   

Fragmentation Errors (Low Risk): Splitting complex International Nonproprietary Names (INNs) or monoclonal antibodies (-mab, -zumab) into separate words, such as transcribing "adalimumab" as "a dalimumab", which confuses downstream NER taggers.   

Non-Word Phonetic Hallucinations (Low Risk): Inventing plausible-sounding gibberish approximations when encountering complex Latin or Greek stems—such as "rezenquizumab" for "risankizumab", "bactarvy" for "biktarvy", or "high flow nephrosis" for "hydronephrosis".   


To mitigate these four error classes in Stage 2, implement a Three-Step Pharmaceutical Post-Correction Gateway:   

Key-Terms Formulary Prompting: Pre-load a customization layer containing high-frequency clinical vocabulary and expected hospital formulary drug names into the ASR prompt. In vendor benchmarks, combining medical-domain acoustic models with key-terms prompting reduces overall Missed Entity Rates (MER) to as low as 3.2% and catches roughly 20% more medical entities than general models.   

Character-Aware Misspelling Detection (CharBERTDrug + RxNorm): Run a token-level misspelling detector modeled on CharBERTDrug and BERTDrug—developed in a September 2026 JMIR Medical Informatics study by fine-tuning CharacterBERTmedical and BERTmedical on synthetic perturbations (insertion, deletion, substitution, transposition, and phonetic shifts) across 69,824 RxNorm drug strings (which average 2.4 words [SD 1.5] and 19.0 characters [SD 12.1]). When a candidate drug string is flagged as misspelled or out-of-vocabulary, retrieve top-k candidate strings via character-3-gram approximate nearest-neighbor lookup against scispaCy's rxnorm linker (~100,000 normalized clinical drug concepts).   

Contextual LLM Validation Gateway (Temperature 0.2–0.4): Pass the candidate drug token, the full conversational turn, and the target RxNorm formulary candidates to an LLM validation gateway configured at a temperature between 0.2 and 0.4. This narrow temperature band prevents creative hallucination while allowing the LLM to use surrounding clinical context to resolve phonetic ambiguities (such as correcting omeprizole to omeprazole, metforman to metformin, or flagging a sound-alike substitution that conflicts with the patient's diagnosis).   

Stage 3: Clinical NLP Pre-Extraction and Constrained SOAP Note Generation
Deterministic Clinical NLP Pre-Extraction (Med7, scispaCy, and negspaCy)
A major vulnerability in basic AI scribes is relying solely on a generative LLM to identify clinical entities and their negation states from raw dialogue. Before invoking the SOAP generator LLM, your pipeline should run a fast, deterministic spaCy Clinical Information Extraction Pipeline over the corrected transcript to build a structured EncounterFactSheet JSON object. This pre-extraction layer combines five specialized components:   

Posology and Medication Attribute Extraction (Med7): Load the en_core_med7_lg (word-vector CNN) or en_core_med7_trf (RoBERTa-base transformer) model trained on MIMIC-III electronic health records. Med7 recognizes seven medication-related categories: DRUG, STRENGTH (e.g., 400mg/5ml), DOSAGE (e.g., 30ml), FREQUENCY (e.g., bid), ROUTE (e.g., PO, intravenously), FORM (e.g., suspension), and DURATION (e.g., for the next 5 days). On MIMIC-III / n2c2 test sets, Med7 achieves a strict micro-averaged F1 of 0.893 and a lenient micro-F1 of 0.957 (with lenient F1 reaching 0.971 for DRUG, 0.979 for FREQUENCY, and 0.964 for ROUTE). Link downstream attributes (STRENGTH, DOSAGE, FREQUENCY, DURATION) to the closest DRUG entity within the same sentence or preceding two sentences.   

Biomedical Entity Recognition and Zero-Shot Generalization (scispaCy & GLiNER): Run en_core_sci_scibert (~785k vocabulary backed by allenai/scibert-base, achieving 92.12% UAS and 90.58% LAS dependency parsing accuracy) alongside en_ner_bc5cdr_md to extract DISEASE and CHEMICAL mentions. To extract custom clinical categories beyond fixed taxonomies, integrate GLiNER, an open-source zero-shot biomedical NER framework.   

Abbreviation Expansion (AbbreviationDetector): Clinical transcripts contain dense shorthand that can confuse downstream verification. Insert scispaCy's AbbreviationDetector component (implementing the Schwartz & Hearst algorithm) into the spaCy pipeline (nlp.add_pipe("abbreviation_detector", config={"make_serializable": True})) so abbreviations are automatically linked to their full definitions prior to ontology resolution.   

Clinical Negation and Speculation Detection (negspaCy / NegEx): In clinical documentation, a pertinent negative (e.g., "Patient has a headache, but no fever") is diagnostically vital. Add negspaCy (nlp.add_pipe("negex", config={"neg_termset": termset("en_clinical_sensitive").get_patterns(), "chunk_prefix": ["no"]})) to evaluate four NegEx pattern classes: preceding negations (e.g., "denies", "no sign of"), following negations (e.g., "was negative", "absent", "declined"), pseudo-negations (double negatives or false triggers like "not ruled out" or "might not"), and termination tokens (conjunctions like "but" or "however" that cut a sentence's negation scope). Including "chunk_prefix": ["no"] is critical because biomedical NER models like en_core_sci_sm often chunk "no" inside the noun phrase (e.g., "no headache" or "no bleeding"), which would otherwise cause NegEx to miss the preceding negation. In clinical NLP benchmarks, combining a customized rule-based NegEx layer with a neural CNN classifier achieved 1.00 precision, 0.99 recall, and 0.99 F1 on non-affirmative clinical entities.   

Ontology Concept Normalization (UmlsEntityLinker): Attach scispaCy's EntityLinker (nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})) to link extracted entities via character-3-gram approximate nearest-neighbor search to five supported biomedical knowledge bases: UMLS (~3M concepts), RxNorm (~100k clinical drug concepts), MeSH (~30k headings), Human Phenotype Ontology (HPO) (16k phenotypic abnormalities), and Gene Ontology (GO) (67k gene concepts).   

Selecting and Fine-Tuning the Open-Weight Medical LLM
For a hospital-deployable project that keeps Protected Health Information (PHI) on-premise—satisfying both HIPAA and India's Digital Personal Data Protection (DPDP) Act—you should prioritize self-hosted, open-weight medical LLMs. Table 2 synthesizes the 2026 open-weight healthcare LLM landscape reported in Nirmitee's 2026 clinical model guide and academic benchmarks so you can justify your backbone selection in your CSET343 report.   

Model Name	Parameter Sizes	Base Architecture & Context	MedQA Score	PubMedQA Score	Multimodal	License	Recommended Role in CARE'OS Module 2
MedGemma 1.5 (Google, Jan 2026)	4B / 27B	Gemma 3	~91%	~81%	Yes (Text + Medical Imaging)	Gemma License (Commercial permitted)	
Primary SOAP Generator & Slow-Path Judge: Leading open-weight medical model (surpassing Med-PaLM 2's 86.5%); 27B runs in ~15–18 GB VRAM at 4-bit AWQ/GPTQ quantization on a single RTX 4090 / A10G with 1–3% benchmark trade-off (or ~60 GB VRAM in BF16).

OpenBioLLM	8B / 70B	Llama 3 (8K context)	~73%	~78%	No	Apache 2.0	
Commercial-Grade Alternative: Strong modern Llama 3 architecture with zero commercial licensing restrictions.

Llama 3 (General Base)	8B / 70B / 405B	Llama 3	~72%	~73%	Vision in 3.2	Llama 3 Community	
Fine-Tuning Substrate: QLoRA fine-tuning of Llama3-8B (LoRA rank 64, lr 0.0002 on an RTX 4090) outperforms standalone GPT-4 on clinical summarization metrics.

Meditron (EPFL / Yale)	7B / 70B	Llama 2 (4K context)	~70%	~74%	No	Llama 2	
Guideline Verification: Trained on WHO, FDA, EMA, and NICE clinical guidelines; restricted by its 4K context window for long encounters.

PMC-LLaMA	7B / 13B	Llama 2	~62%	~72%	No	Llama 2	
Academic Baseline: Useful as a historical benchmark comparison in your report.

BioMistral	7B	Mistral 7B	~60%	~71%	No	Apache 2.0	
Fast First-Pass Extractor: Runs on a single 16 GB consumer GPU at 4-bit quantization; achieves 80.53 BERTScore F1 and 19.61 MEDCON on progress note generation.

  
Section-by-Section Constrained Decoding (Pydantic + Instructor)
When prompting MedGemma 1.5 (or its SOAP-tuned variant Ephraimmm/medgemma-soap-finetuned1) to generate the clinical note, combine RAG (dynamically injecting the extracted EncounterFactSheet into the prompt for factual accuracy) with LoRA/QLoRA fine-tuning (which requires 1,000–5,000 curated examples to lock in institutional SOAP formatting), and enforce three architectural rules:   

Section-by-Section Modular Generation: As established in the ACI-Bench benchmark, generating the clinical note one section at a time (Subjective [Chief Complaint, HPI, Review of Systems, Medications], Objective Exam, Objective Results, and Assessment & Plan) beats generating the full note in a single pass.   

Disable Unconstrained Reasoning During Initial Drafting: Recent 2026 empirical work on ambient clinical intelligence (arXiv:2609.22239) notes that explicitly enabling open-ended reasoning during initial draft generation can degrade generated note quality; instead, draft at a low temperature (0.1) to retain reference fidelity, and reserve structured multi-step reasoning for the 2nd-Pass Verification and Revision stage.   

Schema-Enforced Constrained Decoding: Never parse LLM markdown output by hand. Wrap your LLM call with Instructor and Pydantic, which translates your Python BaseModel into a strict JSON schema enforced via constrained token decoding and automatic validation retries. Require every generated sentence in Subjective, Objective, Assessment, and Plan to be emitted as an AtomicClinicalClaim JSON object containing claim_id, soap_section ("S", "O", "A", or "P"), claim_text, source_utterance_ids (the transcript turn IDs supporting the statement), audio_start_sec and audio_end_sec (inherited from WhisperX word timestamps), and extracted_cuis (UMLS/RxNorm codes from scispaCy).   

Stage 4: The Dual-Tier 2nd-Pass Verifier Engine
The 2nd-Pass Verifier is the core academic and safety contribution of your module. In a clinical note, hallucinations are rarely dramatic fabrications; they are subtle, high-risk distortions—such as an invented medication dosage, a boilerplate normal physical exam finding that was never performed, an unmentioned date, or a diagnosis that contradicts a documented finding.   

The Fundamental Flaw of Literal Verification (Lexical vs. Inference-Aware Evaluation)
An April 2026 study, Beyond Literal Summarization: Redefining Hallucination for Medical SOAP Note Evaluation (Vachhani et al., arXiv:2604.14829), demonstrated why standard hallucination detectors fail on clinical notes. Unlike standard summarization, writing a SOAP note requires clinical abstraction, normalization of colloquial lay language, and medically grounded inference.   

When an Inference-Unaware / Lexical Judge (Stage 1) evaluates a SOAP note by requiring strict, explicit literal grounding in the transcript, it produces a mean hallucination rate of 35.2% across 100 physician-patient encounters. However, when human physician annotators evaluate the exact same notes, they measure an average hallucination rate of 10.4%. The 24.8 percentage-point gap consists of valid clinical reasoning and terminology normalizations falsely penalized by literal matching—such as upgrading "throat irritation" to "pharyngeal sensitivity", mapping the trade name "Zestril" to generic "lisinopril", or inferring "Peptic Ulcer Disease / GERD" from a cluster of epigastric burning, nocturnal awakening, and six months of daily ibuprofen use. By replacing literal matching with an Inference-Aware, Ontology-Grounded Judge (Stage 2), the mean hallucination rate drops to 9.1%—bringing automated verification within 1.3 percentage points of human physician judgment while maintaining complete agreement with Stage 1 on true patient-safety fabrications (Tiers 4–5).   


Architectural Walkthrough of Your Dual-Tier 2nd-Pass Verifier
To achieve both low latency and clinical reasoning depth, architect your 2nd-Pass Verifier as a cascaded two-tier system (Fast-Path NLI Filter → Slow-Path Ontology-RAG Judge):   

Step 1: Atomic Fact and Claim Decomposition (FActScore / FactEHR Protocol)
Long-form clinical sentences frequently contain a mixture of supported and unsupported clauses, making binary sentence-level judgment inadequate. Following the FActScore (Factual Precision in Atomicity Score) and FactEHR frameworks, your verifier first extracts the set of stated atomic facts from the transcript (capturing symptoms, medications, durations, history, and physician observations) and breaks the draft SOAP note into atomic claims (a typical clinical generation contains 26 to 41+ atomic facts across ~49 sentences).   

Step 2: Tier-1 Fast-Path NLI Screening (MiniCheck & DeBERTa-v3 MedNLI)
Running a 27B LLM judge over 40+ atomic claims sequentially is computationally expensive. Instead, pass all extracted fact-claim pairs in a single GPU batch through two specialized 0.4B-parameter DeBERTa-v3 cross-encoders:

lytang/MiniCheck-DeBERTa-v3-Large: Fine-tuned from microsoft/deberta-v3-large on 35,000 examples (21,000 ANLI pairs + 14,000 structured synthetic fact-checking claims), MiniCheck processes approximately 800 documents per minute and outputs a binary grounding label (1 = supported, 0 = unsupported) alongside a raw confidence probability (raw_prob).   

shidey/deberta-v3-mednli-scifact-open-sentence-nli: Built via a three-stage sequential transfer learning chain—starting from MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (885K general NLI pairs) → fine-tuned on 11,232 clinical pairs from MedNLI → fine-tuned on 728 scientific pairs from SciFact-Open using inverse-frequency class-balanced cross-entropy loss:

weight 
c
​
 = 
K×n 
c
​
 
N
​
 

where N = 728 total samples, K = 3 classes, and n_c represents the samples per class. When direct evidence sentences are not explicitly linked, the pipeline constructs the premise by retrieving the closest transcript sentence by embedding cosine similarity using abhinand/MedEmbed-large-v0.1. This model classifies each claim into LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}, achieving 87.0% overall clinical accuracy and 89.0% contradiction recall on the 1,422-pair MedNLI test set.   

Bi-Directional Entailment & Fast-Path Routing: To maximize sensitivity on subtle clinical distortions, compute the bi-directional entailment score used in the MedHallu benchmark:
  

E(c 
i
​
 ,F 
i
​
 )=min(NLI(c 
i
​
 →F 
i
​
 ), NLI(F 
i
​
 →c 
i
​
 ))

When MiniCheck predicts 1 with high confidence and MedNLI predicts entailment (ID 0), the claim is immediately certified as Tier 1: Verbatim / Directly Supported, bypassing heavy LLM compute. Conversely, if MedNLI predicts contradiction (ID 2) or if negspaCy detects a polarity flip (where a negated transcript entity with negex=True is asserted affirmatively in the note), the claim is immediately flagged as a Tier 5: Direct Contradiction. All claims classified as neutral (ID 1) or unsupported (0) by MiniCheck are routed to the Tier-2 Slow-Path Ontology-RAG Judge to distinguish valid clinical inferences (Tiers 2a, 2b, 3) from true fabrications (Tier 4).   

Step 3: Tier-2 Slow-Path Ontology-RAG Judge & The 5-Step CoT Protocol
When a claim is not literally stated in the transcript, the Slow-Path Judge (MedGemma 1.5 27B or a local Qwen2.5-72B FactEHR judge) queries a structured medical knowledge base drawing on SNOMED CT, ICD-10, and open-source pharmacological databases (RxNorm), and executes the Five-Step Chain-of-Thought (CoT) Verification Protocol:   

State the Claim: Isolate the individual atomic assertion from the SOAP note.   

Cross-Section Fact Scan & Non-Negotiable Synonym Rule: Scan extracted facts across all four SOAP sections (Subjective, Objective, Assessment, and Plan) rather than only the matching section (for instance, connecting an allergy mentioned in Subjective to a medication choice in Plan). Apply the Non-Negotiable Synonym Rule: trade names and generic drug names (e.g., Zestril vs. lisinopril) are treated as strictly equivalent under all conditions.   

Evaluate Direct/Synonym Match: If a direct mention, synonym, translated equivalent, or paraphrase is found → mark Supported (Tier 1 / Tier 2b) and stop.   

Evaluate Clinical Abstraction and Inference: If not found literally, ask: Is this a medically reasonable inference from symptoms/history/findings, a standard clinical summary statement appropriate given context (e.g., "hemodynamically stable" or "in no acute distress" when vitals are normal), or a medical terminology upgrade (Tier 2a)? If yes → mark Supported (Tier 2a / Tier 3) and stop.   

Enforce the Safety Floor (Tiers 4–5): Otherwise, mark Hallucinated (Tier 4–5) and state the specific safety violation reason.   

Table 3 details the complete 5-Tier Clinical Claim Taxonomy and the automated remediation rules (Repair-Trim-Withhold) that your system should enforce.   

Verification Tier	Clinical Category	Concrete Examples from Clinical Benchmarks	Verifier Verdict	Automated Remediation Action (Repair-Trim-Withhold)
Tier 1	Direct Verbatim Mention	
Transcript: "Blood pressure is 124/82." → Note: "BP 124/82 mmHg."

Supported (Green)	
Retain claim; attach exact WhisperX audio timestamp link.

Tier 2a	Lay-to-Medical Terminology Upgrade	
Transcript: "Pain after I eat" → Note: "Postprandial pain"; "Throat irritation" → "Pharyngeal sensitivity".

Supported: Terminology Upgrade (Green)	
Retain claim; display original lay phrase from transcript on hover tooltip.

Tier 2b	Ontology Synonym / Drug Equivalence	
Transcript: "Taking Zestril daily" → Note: "Continue lisinopril daily".

Supported: RxNorm Synonym (Green)	
Retain claim; append RxNorm CUI and brand/generic mapping in metadata.

Tier 3	Clinically Grounded & Cross-Section Inference	
Epigastric burning + nocturnal awakening + 6 mos daily ibuprofen → Assessment: "Suspected PUD / GERD"; or linking Subjective allergy to Plan drug.

Supported: Clinical Inference (Amber Badge)	
Retain claim with an [AI-Inferred] badge requiring a 1-click physician confirmation check before EHR sign-off.

Tier 4	Unsupported Extrapolation / Boilerplate Fabrication	
Unmentioned physical exam ("Neck supple without thyromegaly"), invented vitals/dates ("PSA 0.6 ng/mL, 05/25/2021"), or unmentioned dosage/route.

Hallucinated: Unsupported Fact (Red Alert)	
Auto-Trim or Withhold: Strip unsupported exam/date spans automatically (as done in ACI-Bench curation) or lock behind mandatory clinician override.

Tier 5	Direct Contradiction & Polarity Flip	
Transcript: "Patient denies fever" → Note: "Presents with fever"; or documenting a diagnosis contradicted by lab results.

Hallucinated: Safety Contradiction (Critical Red)	
Auto-Repair: Trigger targeted regeneration conditioned on the MedNLI contradiction span and negspaCy polarity tag.

  
Step 4: Internal Clinical Logic & Medical Error Check (MEDEC Verification Layer)
Even if a note faithfully reflects what was spoken, the draft note or the clinician's hurried dictation might contain an internal clinical error. Add a final check modeled on the 2025 MEDEC (Medical Error Detection and Correction) benchmark (3,848 clinical texts across MS and UW hospital collections). This module scans the completed SOAP note across the five MEDEC error categories—(1) Diagnosis, (2) Management, (3) Treatment, (4) Pharmacotherapy, and (5) Causal Organism—executing three sub-tasks: predicting the binary error flag (0 or 1), extracting the exact erroneous sentence, and generating a corrected sentence. In addition, include a rule-based filter that flags Joint Commission "Do Not Use" ambiguous medical abbreviations or missing numerical units in the Objective and Plan sections.   

Five Novel Features to Make Your CSET343 Project Stand Out
To elevate your project above standard university submissions and showcase deep domain mastery, integrate the following five novel features into your CARE'OS Module 2 implementation.

1. Code-Switched "Hinglish" Ambient Scribe Engine for Indian Outpatient Clinics
Western ambient AI scribes struggle in Indian hospitals because Indian clinical communication is structurally code-switched. Physicians trained in English medical colleges naturally blend English clinical terminology with Hindi (or regional Indic) syntax and auxiliary verbs within a single sentence—for example: "Aapko kitne din se yeh problem hai? Dyspnea on exertion ke saath chest tightness bhi feel ho raha hai?" or "Patient ko fever hai with associated chills aur rigors". Standard multilingual Whisper models fail on this input for two technical reasons: (1) their acoustic encoders assume monolingual phoneme inventories and force non-English words into English phonetic approximations, and (2) language-identification classifiers operate on 1-to-3-second audio windows that cannot switch fast enough when a single Hindi word (such as "hai") is embedded mid-utterance between English medical terms. Furthermore, pure Devanagari Hindi transcription can be brittle for downstream clinical LLMs because minor matra or nukta errors alter word meanings completely (for example, बाल [baal = hair] vs. बल [bal = strength], or दिन [din = day] vs. दीन [deen = poor/humble]).   

To implement an "Indian Clinic Mode" in Module 2, integrate three recent open-source Hinglish speech techniques. First, deploy Trelis/whisper-hinglish-preview (a Whisper-large-v3 model fine-tuned from IISc ARTPARK's whisper-large-v3-vaani-hindi) and inject the special <|mixedcode|> token right after the language token in the decoder prompt: prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]. On internal evaluations using the CoSHE-500 conversational code-switching benchmark, this model reduces WER to 13.67% compared to 29.74% for base whisper-large-v3 and 73.96% for Vaani (and achieves 12.73% WER on hiacc-adult accented code-switched speech vs. 28.53% for base whisper-large-v3). Second, offer a Roman-script fallback using Oriserve/Whisper-Hindi2Hinglish-Prime (fine-tuned on 550 hours of noisy Indian-accented audio under an Apache 2.0 license), which reports a ~39% average WER reduction across three public datasets and cuts Common Voice WER from 61.94% to 32.43% while outputting Latin-script Hinglish that integrates directly with English clinical LLMs. Third, replicate the 1st-place cascaded architecture from the 2026 DISPLACE-M Hinglish Medical Conditions Extraction Challenge (arXiv:2603.06373) by conditioning your Speaker-Attributed ASR (SA-ASR) exclusively on active speech segments identified by the EEND-VC / DiariZen diarizer—preventing ambient clinic noise from triggering ASR hallucinations—and extend your 2nd-Pass Verifier's Tier 2a rule to validate Hindi-to-English symptom translations as supported clinical abstractions.   

2. Interactive "Linked Evidence" Audio-Provenance and Sub-Second Audit UI
In enterprise health-system deployments (such as Abridge's rollout across more than 100 health systems including Kaiser Permanente, Johns Hopkins, Duke, and Mayo Clinic), the product differentiator that builds clinician trust is auditability at the line level via "Linked Evidence". Clinicians are reluctant to sign AI-generated notes if they must manually scan a long transcript to verify where a specific medication or history element originated.   

Using the word-level phoneme forced-alignment timestamps generated by WhisperX in Stage 1 and the source_utterance_ids enforced by your Pydantic schema in Stage 3, build a split-screen Linked Evidence Web Editor. When the physician highlights or clicks any sentence in the SOAP note editor on the right panel, the UI automatically highlights the corresponding passage in the diarized transcript on the left panel and cues the exact audio slice ([audio_start_sec, audio_end_sec]) so the clinician can double-click or press Play to listen to the original spoken recording. Pair this with inline color-coded verification badges from Stage 4 (solid green for Tier 1/2a/2b Supported Claims, amber [AI-Inferred] pills for Tier 3 Clinical Inferences, and red warning/strikethrough badges for Tier 4/5 Unsupported or Contradicted Claims) and a case-insensitive Transcript Search bar at the bottom of the transcript modal so clinicians can jump directly to keywords like medications, procedures, or social history.   


3. Real-Time "Clinical Co-Pilot" Whisperer for Differential Diagnosis and Red-Flag Nudges
Most student AI scribe projects only execute after the consultation ends. However, if a physician forgets to ask a critical red-flag question during the encounter, even a perfect scribe can only document an incomplete history. A September 2026 case-level randomized, adjudicator-blinded simulation study published in JMIR Formative Research (evaluating 13 board-certified family and community medicine physicians across 260 simulated voice-based primary care consultations) demonstrated that a real-time AI diagnostic copilot increased Top-3 diagnostic accuracy from 62.3% (95% CI 49.1%–75.7%) unassisted to 74.6% (95% CI 63.7%–85.4%) with AI assistance—an adjusted absolute improvement of +12.3 percentage points (Adjusted Odds Ratio [AOR] 2.68, P = .01; simulation Number Needed to Treat [NNT] equivalent = 8.1). While Top-1 accuracy point estimates favored AI assistance without reaching statistical significance (57.0% vs. 47.8%, AOR 1.83, P = .08), exploratory interaction analyses revealed that on the highest-difficulty quartile of clinical cases, AI copilot assistance increased diagnostic accuracy by +43.9 percentage points (and +31.1 percentage points in Quartile 2, P < .001).   

To implement this in Module 2, add an asynchronous Real-Time Co-Pilot Agent (modeled on DeepMind's dual-agent Planner + Talker Co-Clinician safeguard architecture and the CLARITY clinical dialogue system) that monitors rolling transcript chunks during the active visit:   

Dynamic Differential Hypothesis Tracker: Periodically query the LLM to maintain the top-N working diagnostic hypotheses based on the patient's complaints and medical history gathered so far, repeating the query if fewer than N hypotheses are returned.   

Vector-Filtered Red-Flag Question Generator: Following the CLARITY inquiry generation protocol, first search a historical clinical dialogue vector database; if a similar validated case is found with cosine similarity > 0.965, reuse its curated follow-up questions. If no match is found, prompt the LLM to generate five candidate follow-up / red-flag questions, compare each candidate against prior dialogue questions in the current session, and discard any candidate with cosine similarity > 0.86 to prevent redundant prompting.   

Anti-Overreliance UI Design: In the 2026 JMIR Formative Research trial, when physicians were presented with risk scenarios involving an incorrect AI suggestion, concordance with the flawed suggestion was 55.8% in the AI-assisted group versus 38.8% unassisted (AOR 2.08, P = .09). To mitigate clinician overreliance and anchoring bias, design your Co-Pilot drawer to surface "Unasked Red-Flag Questions & Pertinent Negatives to Rule Out" rather than authoritative single diagnoses.   

4. Bidirectional CARE'OS Inter-Module Bus (Connecting Modules 1, 2, and 3)
Because your course project is an integrated operating system (CARE'OS), demonstrating automated data orchestration across all three modules will make your architecture stand out immediately:

A. Module 2 → Module 3: Automated CPIC Level A Pharmacogenomic (PGx) Trigger Bus
In routine clinical practice, pharmacogenomic testing is underutilized because genetic test results are rarely linked dynamically to medication prescribing inside the EHR workflow. A large-scale geriatric study (PMC12748018) found that 43.4% of older adults (433.8 per 1,000 individuals) were prescribed at least one medication with an actionable CPIC Level A pharmacogenetic biomarker (predominantly CYP2C19- and SLCO1B1-associated drugs such as clopidogrel, PPIs, and statins), and an estimated 36.5% of exposed patients require genotype-guided dose adjustments to mitigate adverse drug reactions or optimize efficacy.   

In your system, the moment Med7 and scispaCy extract a DRUG entity in the draft Plan section and resolve its RxNorm CUI, Module 2 checks the medication against an embedded CPIC Level A / PharmGKB Level 1A Watchlist (covering 30+ peer-reviewed gene-drug pairs). When matched, Module 2 sends an asynchronous JSON request ({patient_id, rxnorm_cui, drug_name, prescribed_dose, target_genes}) to Module 3 (Genomics-Informed Treatment Recommendation). Module 3 evaluates the patient's genotype and returns a structured prescribing alert directly into Module 2's 2nd-Pass Verifier Console before the physician signs the note. Table 4 provides the lookup table of high-priority CPIC Level A drug-gene triggers to build into your Module 2 ↔ Module 3 bridge.   

Prescribed Drug (Extracted by Med7 / RxNorm)	Therapeutic Class	Target Gene(s) Sent to Module 3	High-Risk Phenotype / Allele Example	CPIC Level A Clinical Action Displayed in Module 2 Verifier UI
Clopidogrel (Plavix)	Antiplatelet	CYP2C19	Poor / Intermediate Metabolizer	
Critical Alert: Reduced platelet inhibition and increased risk for adverse cardiovascular events. Recommend alternative antiplatelet therapy (prasugrel or ticagrelor) if no contraindication.

Warfarin (Coumadin)	Anticoagulant	CYP2C9, VKORC1, CYP4F2, rs12777823	CYP2C9 *2/*3 or *5, *6, *8, *11 + VKORC1 -1639G>A (A/A)	
Dosing Alert: CYP2C9 *2 and *3 impair S-warfarin metabolism by ~30–40% and ~80–90%, respectively; apply genotype-guided dose reduction or consider alternative oral anticoagulant.

Simvastatin / Atorvastatin / Rosuvastatin / Fluvastatin	Statins	SLCO1B1, ABCG2 (rs2231142), CYP2C9	Decreased SLCO1B1 / ABCG2 function	
Myopathy Risk Alert: High prevalence exposure (131.1/1,000 for atorvastatin; 56.1/1,000 for simvastatin); adjust statin dose or switch agent to reduce toxicity risk.

Codeine / Tramadol	Opioid Analgesics	CYP2D6 (OPRM1, COMT)	Ultra-Rapid Metabolizer (UM) or Poor Metabolizer (PM)	
Toxicity / Inefficacy Alert: Codeine is metabolized by CYP2D6 to morphine; UMs face severe toxicity risk, while PMs experience reduced morphine formation and no pain relief. Avoid codeine and tramadol.

Allopurinol	Gout Therapy	HLA-B*58:01	HLA-B*58:01 Positive	
Safety Alert: Actionable CPIC Level A gene-drug interaction; screen HLA-B*58:01 status to guide alternative therapy.

Carbamazepine / Oxcarbazepine / Phenytoin	Anticonvulsants	HLA-B*15:02, HLA-A*31:01, CYP2C9	High-risk HLA-B*15:02 / HLA-A*31:01 / CYP2C9 variants	
Safety Alert: Actionable CPIC Level A anticonvulsant interaction requiring genotype-guided drug selection or dosing.

Omeprazole / Pantoprazole / Lansoprazole	Proton Pump Inhibitors (PPIs)	CYP2C19	CYP2C19 metabolizer variants	
Dosing & Interaction Alert: Adjust PPI dosing per CYP2C19 status; warn that clopidogrel cannot be co-administered with omeprazole (Prilosec) or esomeprazole (Nexium).

Azathioprine / Mercaptopurine / Thioguanine	Thiopurine Immunosuppressants	TPMT, NUDT15	Deficient TPMT or NUDT15 status	
Dosing Alert: Apply CPIC thiopurine dosing guidelines based on TPMT and NUDT15 genotypes.

Capecitabine / Fluorouracil / Tegafur	Oncology (Fluoropyrimidines)	DPYD	DPYD deficiency variants	
Toxicity Alert: Apply CPIC DPYD guideline dose modifications for fluoropyrimidines.

Gentamicin / Tobramycin / Amikacin	Aminoglycoside Antibiotics	MT-RNR1	MT-RNR1 risk variants	
Safety Alert: Actionable CPIC Level A antibiotic-gene association (MT-RNR1).

  
B. Module 2 ↔ Module 1: Live Consultation Progress & Queue Pacing Telemetry
Static queue predictors (Module 1) degrade when an outpatient consultation runs significantly longer or shorter than scheduled. Because your Module 2 scribe tracks the live dialogue, it can monitor which stage of the visit is currently unfolding (Subjective History → Objective Exam Dictation → Assessment & Plan) and count the active clinical problems (UMLS concepts) discussed. Design Module 2 to emit a periodic JSON telemetry heartbeat to Module 1 containing the elapsed duration, current detected SOAP phase, active extracted concept count, and speaker overlap ratio so Module 1 can dynamically update predicted wait times for patients in the waiting room. Conversely, during the Pre-Visit phase, Module 1 can pass the patient's chief complaint and prior visit context into Module 2 to initialize the ASR vocabulary prompt and Co-Pilot context before the patient enters the room.   

5. Multi-Audience Post-Visit Artifact Generator and HL7 FHIR R4 Exporter
Once the physician reviews the 2nd-Pass Verifier badges and signs the note, Module 2 should automatically compile three downstream clinical artifacts:   

HL7 FHIR R4 / USCDI Interoperability Bundle: Map the verified Pydantic SOAP schema directly into standardized HL7 FHIR Observation, Condition, and ServiceRequest/MedicationRequest JSON objects complete with USCDI data classes, SNOMED CT / ICD-10 diagnosis codes, and RxNorm medication identifiers.   

Automated E/M Coding & Medical Necessity Readiness Check: Verify that the finalized note meets Evaluation and Management (E/M) documentation expectations for medical necessity and clinical complexity (such as G2211 complexity add-on criteria, explicit differentials, monitoring parameters, and task assignments).   

Bilingual Patient-Friendly After-Visit Summary (AVS) with Teach-Back Prompts: Translate the verified Assessment and Plan into plain, jargon-free language (in English, Hindi, or Hinglish) tailored to patient health literacy, complete with explicit Red-Flag Warning Signs / Emergency Escalation Steps and a Teach-Back Verification Box that helps clinicians confirm patient comprehension before discharge.   

Datasets, Evaluation Metrics, and Experimental Design for Your Project Report
To earn top marks in CSET343, your project report should present a reproducible quantitative evaluation across public clinical benchmarks.

Public Datasets to Train, Test, and Demo Module 2
You do not need private hospital data to build and benchmark your pipeline. Table 5 details six open-access datasets you can combine to train, test, and demonstrate every stage of Module 2.   

Dataset Name	Modality & Size	Average Length (Source → Target)	Primary Use in Your Module 2 Project	Access & Repository
ACI-Bench (Yim et al., Nature Sci. Data 2023)	
207 full role-played & virtual scribe encounters (67 train, 20 valid, 120 test across 3 test sets of 40).

1,302 tokens / 55 turns dialogue → 490 tokens / 49 sentences structured visit note.

Primary End-to-End Benchmark: Evaluate full-note and section-by-section SOAP generation across Virtual Assistant (virtassist), Virtual Scribe (virtscribe), and Ambient (aci) subsets.

Open (CC BY 4.0) on GitHub (wyim/aci-bench) & HuggingFace.

PriMock57 (Papadopoulos Korfiatis et al., ACL 2022)	
57 full primary care mock consultations with raw audio, manual utterance transcripts, notes, and eval checklists.

1,489 tokens / 97 turns dialogue → 161 tokens / 23 sentences consultation note.

Audio-to-Note & Diarization Testing: Test WhisperX + pyannote word alignment and Linked Evidence audio timestamp playback on real consultation audio.

Open on GitHub (babylonhealth/primock57 via Git LFS) & HuggingFace.

MTS-Dialog (Ben Abacha et al., EACL 2023)	
1,701 doctor-patient dialogue-note snippets (100 val, two test sets of 200) across 20 normalized section headers (GENHX, MEDICATIONS, CC, PASTMEDICALHX, etc.).

142 tokens / 9 turns dialogue → 48 tokens / 3 sentences section summary (plus French/Spanish augmented sets).

Section-Level Fine-Tuning & Fast Testing: Fine-tune Llama3-8B or MedGemma-4B via QLoRA on localized section extraction and test Med7 / negspaCy.

Open on GitHub (abachaa/MTS-Dialog) & HuggingFace (har1/MTS_Dialogue-Clinical_Note).

DISPLACE-M & CoSHE-500 (2026)	
~35 hours (125 dev dialogues = 24.5h, 30.1k turns; 10.13h blind eval) of rural Indian ASHA/clinician-patient Hinglish recordings; 500 conversational CS clips.

241.2 turns (~11 min 47 sec) per session; 84.30% single-speaker, 11.35% silence, 4.35% overlapping speech.

Hinglish ASR & Overlap Diarization Benchmark: Evaluate your Indian Clinic code-switched pipeline (whisper-hinglish-preview + DiariZen).

Public via DISPLACE-M challenge & HuggingFace (soketlabs/CoSHE-Eval).

MEDEC (Ben Abacha et al., ACL Findings 2025)	
3,848 clinical texts (2,189 train, 734 val, 925 test across MS and UW hospital systems; including 488 unseen hospital notes) annotated with 5 error types.

Clinical notes where each text is either correct (0) or contains exactly 1 clinical error (1).

2nd-Pass Verifier Error Correction Benchmark: Benchmark your verifier on Task A (Error Flag), Task B (Error Sentence Extraction), and Task C (Correction Generation).

MEDEC-MS open on GitHub (abachaa/MEDEC); UW via DUA.

MedHallu (Pandit et al., EMNLP 2025)	
10,000 medical QA entries derived from PubMedQA categorized into Easy, Medium, and Hard hallucinations (76% question misinterpretation, 0.5% evidence fabrication).

Controlled hallucinated answers within ±10% of ground-truth length density.

Verifier Calibration & Abstention Testing: Test bi-directional NLI scoring and demonstrate how introducing a "not sure" / [Inferred] category improves precision and F1 by up to 38% over baselines.

Open on HuggingFace & Kaggle (medical-hallucination-dataset-medhallu).

  
Multi-Dimensional Evaluation Protocol for Your Defense
A common mistake in academic clinical NLP reports is reporting only ROUGE-L or holistic 1-to-5 Likert scores such as the Physician Documentation Quality Instrument (PDQI-9). As a Suki AI clinical evaluation whitepaper demonstrated across 84 paired notes in 4 specialties, holistic Likert tools like PDQI-9 (validated in 2012 on 9 inpatient notes from 3 charts) suffer from a structural mismatch with LLM error phenomenology: a critical medication dosage error can sit inside a well-organized, cohesive note and vanish from the overall holistic score, while human raters exhibit high inter-rater inconsistency on what counts as a hallucination. Therefore, structure your CSET343 evaluation chapter around a Four-Layer Granular Metric Suite:   

Layer 1 — Clinical Speech & Diarization Metrics (Stages 1 & 2): Report Diarization Error Rate (DER %) without forgiveness collars and Speaker Role Identification Accuracy (%) across Doctor vs. Patient turns. Alongside overall WER, report split clinical speech metrics: Medical-Term WER (M-WER), Drug-Name Error Rate, Missed Entity Rate (MER), and Dosage-Event F1 (aligning canonical number-and-unit events so invented or altered doses are penalized).   

Layer 2 — Syntactic, Semantic, and Concept Completeness Metrics (Stage 3): Report syntactic overlap via BLEU and ROUGE-L and contextual embedding similarity via BERTScore F1. Crucially, include MEDCON (UMLS Medical Concept F1)—introduced with ACI-Bench—which extracts UMLS concepts from both gold and predicted notes using QuickUMLS or scispaCy and calculates the binary F1 score between the two concept sets. In recent ACI-Bench and Pitt-Bench studies, adding a post-generation revision pass improved recall across every tested model by +2.19 to +5.03 points on MEDCON and +2.02 to +5.83 points on FactEHR.   

Layer 3 — Atomic Factuality & 2nd-Pass Verifier Metrics (Stage 4): Measure atomic claim precision using FActScore, FactEHR, and AlignScore. Run an ablation study comparing Stage 1 (Literal/Lexical Judge) against Stage 2 (Inference-Aware Ontology-RAG Judge) to demonstrate the reduction of false-positive flags on Tier 2a/2b/3 clinical transformations while catching 100% of Tier 4–5 fabrications, and report detection and correction accuracy across the 5 clinical error classes in MEDEC-MS.   

Layer 4 — Structural Completeness & Clinical Accuracy Screen: Apply the 60-Second Clinical Accuracy Checklist (incorporating PDQI-9 and QNOTE dimensions of clarity, organization, relevance, and completeness) to verify that no SOAP section—particularly the Assessment section, which is most frequently thin in AI-generated notes—suffers from generic phrasing ("Patient doing fine"), mistranscribed numerals, diagnostic drift across visits, or missing follow-up contingencies.   

Recommended Step-by-Step Implementation Roadmap
To execute this cleanly within your CSET343 project timeline, build Module 2 in four modular sprints:

Sprint 1 (Weeks 1–2): Audio-to-Diarized Transcript & Hinglish Pipeline. Set up a Python ≥3.10 virtual environment. Integrate WhisperX (faster-whisper large-v3 + wav2vec2.0 alignment with numba), pyannote/speaker-diarization-community-1, and the Trelis/whisper-hinglish-preview model for Indian clinical speech. Test word-timestamp extraction on consultation audio from PriMock57.   

Sprint 2 (Weeks 3–4): Deterministic Clinical NLP & Constrained SOAP Generator. Build the spaCy pipeline combining en_core_med7_trf, en_core_sci_scibert, AbbreviationDetector, negspaCy (en_clinical_sensitive + chunk_prefix: ["no"]), and UmlsEntityLinker (umls and rxnorm). Connect 4-bit quantized MedGemma 1.5 (or Llama3-8B / BioMistral-7B) with Pydantic and Instructor to generate section-by-section JSON SOAP notes linked to utterance IDs and audio timestamps.   

Sprint 3 (Weeks 5–6): Dual-Tier 2nd-Pass Verifier & MEDEC Safety Gate. Implement the atomic claim splitter, load lytang/MiniCheck-DeBERTa-v3-Large and shidey/deberta-v3-mednli-scifact-open-sentence-nli for Fast-Path screening, and build the Slow-Path 5-Step CoT Ontology-RAG Judge over RxNorm/UMLS to classify claims into Tiers 1–5 and auto-trim/repair Tier 4–5 hallucinations.   

Sprint 4 (Weeks 7–8): Linked Evidence UI, Live Co-Pilot, and CARE'OS Module 1 & 3 Integration. Build the split-screen web console with one-click audio provenance playback, wire up the CPIC Level A webhook to Module 3 and the encounter progress telemetry stream to Module 1, and run your automated evaluation scripts (WER/M-WER, ROUGE-L, BERTScore, MEDCON, and Stage 1 vs. Stage 2 Hallucination Rate) on ACI-Bench and MEDEC-MS for your final report.   


arxiv.org
Redefining Hallucination for Medical SOAP Note Evaluation
Opens in a new window

arxiv.org
Knowledge Graph-Augmented Ambient AI for Clinical Note Generation
Opens in a new window

pmc.ncbi.nlm.nih.gov
Toward Relieving Clinician Burden by Automatically Generating
Opens in a new window

ar5iv.labs.arxiv.org
The Sound of Healthcare: Improving Medical Transcription ASR
Opens in a new window

corti.ai
Engineering medical-grade ASR - Corti.ai
Opens in a new window

assemblyai.com
How accurate is AI transcription for pharmaceutical drug names?
Opens in a new window

github.com
Med7 - kormilitzin - GitHub
Opens in a new window

arxiv.org
Doctor or Patient? Synergizing Diarization and ASR for Code ... - arXiv
Opens in a new window

github.com
WhisperX: Automatic Speech Recognition with Word-level ... - GitHub
Opens in a new window

huggingface.co
pyannote/speaker-diarization-community-1 - Hugging Face
Opens in a new window

isca-archive.org
Adapting Speaker Diarization to Code-Switched Medical
Opens in a new window

andrewvillazon.com
Clinical Natural Language Processing in Python - Andrew Villazon
Opens in a new window

f22labs.com
Why the Instructor Beats OpenAI for Structured JSON Output
Opens in a new window

nirmitee.io
Opens in a new window

pypi.org
scispacy · PyPI
Opens in a new window

huggingface.co
Opens in a new window

huggingface.co
lytang/MiniCheck-DeBERTa-v3-Large - Hugging Face
Opens in a new window

adeptminds.co.in
Solving the Hinglish Problem: Why Western Clinical AI Fails in
Opens in a new window

huggingface.co
Trelis/whisper-hinglish-preview - Hugging Face
Opens in a new window

support.abridge.com
Verify a Note With Linked Evidence - Abridge Support Documentation
Opens in a new window

aclanthology.org
CLARITY: Clinical Assistant for Routing, Inference, and Triage
Opens in a new window

formative.jmir.org
Opens in a new window

pmc.ncbi.nlm.nih.gov
Use of Drugs With Actionable Pharmacogenetic Biomarkers and
Opens in a new window

cpicpgx.org
Guidelines - CPIC
Opens in a new window

trytwofold.com
Opens in a new window

solarishealth.com
AI Copilot in the Doctor's Office: myth, miracle or menace?
Opens in a new window

abridge.com
Pioneering the Science of AI Evaluation - Abridge
Opens in a new window

pristren.com
Pyannote: Add Speaker Diarization to Whisper Transcription - Pristren
Opens in a new window

scribd.com
WhisperX Speaker Diarization Framework | PDF | Deep Learning
Opens in a new window

pmc.ncbi.nlm.nih.gov
Speaker Role Identification in Clinical Conversations - PMC - NIH
Opens in a new window

medinform.jmir.org
Detecting Misspelled Drug Names Using Transformer-Based
Opens in a new window

kaggle.com
Negation Detection Dataset for Clinical/General Te - Kaggle
Opens in a new window

medium.com
Getting Structured Output from the LLM using Pydantic | by Ravi Kumar
Opens in a new window

dev.to
Stop Parsing JSON by Hand: Structured LLM Outputs With Pydantic
Opens in a new window

python.useinstructor.com
Instructor - Multi-Language Library for Structured LLM Outputs
Opens in a new window

github.com
GitHub - abachaa/MEDEC
Opens in a new window

researchgate.net
WhisperX pipeline: (a) Two-pass transcription process with Whisper
Opens in a new window

pyannote.ai
What is Speaker Diarization? - pyannoteAI Speaker Intelligence and
Opens in a new window

scalastic.io
Whisper and Pyannote: The Ultimate Solution for Speech Transcription
Opens in a new window

medical.linguacopilot.in
Lingua CoPilot Medical: AI Clinical Simulation for Medical & Nursing
Opens in a new window

arxiv.org
Doctor or Patient? Synergizing Diarization and ASR for Code ... - arXiv
Opens in a new window

medrxiv.org
Speaker Role Identification in Clinical Conversations | medRxiv
Opens in a new window

omi.health
Best Medical Speech-to-Text APIs for Healthcare (2026) - Omi Health
Opens in a new window

researchgate.net
(PDF) Aci-bench: a Novel Ambient Clinical Intelligence Dataset for
Opens in a new window

support.prodi.gy
Med7 — an information extraction model for clinical natural
Opens in a new window

arxiv.org
Med7: a transferable clinical natural language processing model for
Opens in a new window

medinform.jmir.org
Extracting Drug Names and Associated Attributes From Discharge
Opens in a new window

medium.com
Clinical Biomedical Named Entity Recognition (NLP) Using Scispacy
Opens in a new window

allenai.github.io
scispacy | SpaCy models for biomedical text processing
Opens in a new window

arxiv.org
GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical
Opens in a new window

docs.knowledgator.com
Extract Biomedical Entities from Clinical Reports with GLiNER
Opens in a new window

youtube.com
NLP Negation Detection: Introduction and Python Tutorial (NegEx)
Opens in a new window

pypi.org
negspacy · PyPI
Opens in a new window

github.com
jenojp/negspacy: spaCy pipeline object for negating concepts in text
Opens in a new window

towardsdatascience.com
Clinical Notes - The "Negative" Story | Towards Data Science
Opens in a new window

pmc.ncbi.nlm.nih.gov
Negation recognition in clinical natural language processing using a
Opens in a new window

kaggle.com
Extract entities linked to UMLS - Kaggle
Opens in a new window

wuraolaoyewusi.com
scispaCy EntityLinker Tutorial: UMLS, MeSH, RxNorm, GO and HPO
Opens in a new window

web.stanford.edu
Clinical Text Summarization with LLM-Based Evaluation
Opens in a new window

huggingface.co
Ephraimmm/medgemma-soap-finetuned1 - Hugging Face
Opens in a new window

pmc.ncbi.nlm.nih.gov
Aci-bench: a Novel Ambient Clinical Intelligence Dataset for ... - PMC
Opens in a new window

ai-tldr.dev
Instructor: Pydantic-Based Structured LLM Outputs - AI/TLDR
Opens in a new window

zenvanriel.com
Instructor for Structured LLM Output - Complete Implementation Guide
Opens in a new window

verifychart.ai
AI Scribe Hallucinations in Clinical Notes: Real Examples - VerifyChart
Opens in a new window

pubmed.ncbi.nlm.nih.gov
Adaptive NLI-Driven Claim Verification with Statistical Decision
Opens in a new window

martiansideofthemoon.github.io
FACTSCORE: Fine-grained Atomic Evaluation of Factual Precision
Opens in a new window

arxiv.org
MedHallu: A Comprehensive Benchmark for Detecting Medical
Opens in a new window

aclanthology.org
A Comprehensive Benchmark for Detecting Medical Hallucinations
Opens in a new window

robosathi.com
Hallucination Checks | RoboSathi
Opens in a new window

healthcarediscovery.ai
MEDEC - Healthcare Discovery
Opens in a new window

aclanthology.org
MEDEC: A Benchmark for Medical Error Detection and Correction in
Opens in a new window

oriserve.com
Whisper-Hindi2Hinglish: our open-source Hinglish speech model
Opens in a new window

github.com
OriserveAI/Whisper-Hindi2Hinglish - GitHub
Opens in a new window

arxiv.org
Doctor or Patient? Synergizing Diarization and ASR for Code ... - arXiv
Opens in a new window

abridge.com
Abridge: Generative AI for Clinical Conversations
Opens in a new window

sacra.com
Abridge Linked Evidence builds trust - Sacra
Opens in a new window

beri.net
Abridge — Overview, Features & Use Cases | THE D*AI*LY BRIEF
Opens in a new window

support.abridge.com
Transcript Search - Abridge Support Documentation
Opens in a new window

abridge.com
Key Highlights and Achievements | A Year of Abridge
Opens in a new window

techrxiv.org
Healthcare Copilot: A Modular Framework for Safe and ... - TechRxiv
Opens in a new window

pubmed.ncbi.nlm.nih.gov
Real-Time Artificial Intelligence Diagnostic Copilot in Simulated
Opens in a new window

deepmind.google
AI co-clinician: researching the path toward AI-augmented care
Opens in a new window

pmc.ncbi.nlm.nih.gov
CPIC: Clinical Pharmacogenetics Implementation Consortium of the
Opens in a new window

zitniklab.hms.harvard.edu
Source code for tooluniverse.pharmgkb_tool - Zitnik Lab
Opens in a new window

attodiagnostics.com
Which Medicines Are Covered In Our Pharmacogenomics Test?
Opens in a new window

luriechildrens.org
Pharmacogenomics Gene – Drug Associations
Opens in a new window

genome.gov
Pharmacogenetics
Opens in a new window

livrepository.liverpool.ac.uk
2017 UPDATE - University of Liverpool Repository
Opens in a new window

santenet.ch
Abridge: Evidence, context, and documentation in one workflow
Opens in a new window

cdn.openai.com
AI-based Clinical Decision Support for Primary Care: A Real-World
Opens in a new window

github.com
GitHub - babylonhealth/primock57: Dataset of 57 mock medical
Opens in a new window

github.com
abachaa/MTS-Dialog: A new collection of 1.7k doctor-patient
Opens in a new window

huggingface.co
Ambient Scribe Evaluation datasets - a sarahgebauer Collection
Opens in a new window

github.com
wyim/aci-bench - GitHub
Opens in a new window

arxiv.org
PriMock57: A Dataset Of Primary Care Mock Consultations - arXiv
Opens in a new window

huggingface.co
har1/MTS_Dialogue-Clinical_Note · Datasets at Hugging Face
Opens in a new window

kaggle.com
Medical LLM Hallucination Dataset (MedHallu) - Kaggle
Opens in a new window

suki.ai
PDQI-9 & AI Note Quality: Zero Signal | Suki Whitepaper
Opens in a new window

imageclef.org
ImageCLEFmed Caption - ImageCLEF
Opens in a new window

arxiv.org
ACI-BENCH: a Novel Ambient Clinical Intelligence - Dataset ... - arXiv
Opens in a new window

rocketdiscoverycentre.ca
assessment of ai-generated pediatric rehabilitation soap-note quality
Opens in a new window

github.com
Medical Hallucination in Foundation Models and Their Impact on
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
Opens in a new window
