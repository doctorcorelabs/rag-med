"""
Medical RAG AI Client — v3.0 (Advanced Copilot Integration)

Key improvements over v2:
- Structured XML evidence formatting with clinical ordering
- Granular inline citations via evidence IDs [E1], [E2] resolved to source references
- Two-pass synthesis for detail requests (extraction then narrative)
- Evidence coverage metadata in output
"""

import json
import re
import urllib.request
import urllib.error
import base64
import mimetypes
from pathlib import Path
from typing import Any

from .retriever import (
    _extract_disease_name,
    _extract_topic_intent,
    _is_detail_request,
    _correct_typo,
    MEDICAL_SYNONYMS,
    SECTION_RULES,
)

COPILOT_TOKEN_URL = 'https://api.github.com/copilot_internal/v2/token'
COPILOT_CHAT_URL = 'https://api.githubcopilot.com/chat/completions'

# Debug session NDJSON (see workspace debug-446f67.log); do not log secrets/PII.
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[4] / "debug-446f67.log"


def _agent_debug_log(hypothesis_id: str, message: str, data: dict[str, Any]) -> None:
    import time
    try:
        line = json.dumps(
            {
                "sessionId": "446f67",
                "hypothesisId": hypothesis_id,
                "location": "copilot_client.py:_call_copilot",
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            },
            ensure_ascii=False,
        ) + "\n"
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

CLINICAL_ORDER: dict[str, int] = {
    "Definisi": 0, "Etiologi": 1, "Patogenesis": 2,
    "Anamnesis": 3, "Manifestasi_Klinis": 4, "Pemeriksaan_Fisik": 5,
    "Diagnosis": 6, "Tatalaksana": 7, "Komplikasi": 8, "Prognosis": 9,
    "Ringkasan_Klinis": 10,
}

EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")


def get_copilot_token(github_token: str) -> str:
    req = urllib.request.Request(COPILOT_TOKEN_URL, headers={
        'Authorization': f'token {github_token}',
        'User-Agent': 'GitHubCopilotChat/0.1.0',
        'Accept': 'application/json',
    })

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('token')
    except urllib.error.URLError as e:
        raise Exception(f"Failed to get Copilot token: {e}")


def _format_evidence_structured(evidence: list[dict[str, Any]]) -> str:
    """Format evidence with XML-like tags and numbering for precise LLM referencing."""
    sorted_evidence = sorted(
        evidence,
        key=lambda e: CLINICAL_ORDER.get(e.get("section_category", ""), 99),
    )

    context = ""
    for idx, item in enumerate(sorted_evidence, 1):
        context += f"""
<evidence id="{idx}">
  <source>{item.get('source_name', '')}</source>
  <page>{item.get('page_no', '')}</page>
  <heading>{item.get('heading', '')}</heading>
  <parent_heading>{item.get('parent_heading', '')}</parent_heading>
  <section_type>{item.get('section_category', 'General')}</section_type>
  <content_type>{item.get('content_type', 'prose')}</content_type>
  <content>
{item.get('content', '')}
  </content>
</evidence>
"""
    return context, sorted_evidence


def _resolve_evidence_citations(text: str, evidence: list[dict[str, Any]]) -> str:
    """Convert [E1], [E2] markers to human-readable (Source, Hal N) references."""
    def _replace(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(evidence):
            item = evidence[idx - 1]
            return f"({item.get('source_name', '?')}, Hal {item.get('page_no', '?')})"
        return match.group(0)

    return EVIDENCE_REF_RE.sub(_replace, text)


def _extract_used_evidence_ids(parsed_json: dict[str, Any]) -> list[int]:
    """Scan all markdown sections for [EN] references and collect used evidence IDs."""
    used: set[int] = set()
    for sec in parsed_json.get("sections", []):
        md = sec.get("markdown", "")
        for m in EVIDENCE_REF_RE.finditer(md):
            used.add(int(m.group(1)))
    vlog = parsed_json.get("verification_log", "")
    for m in EVIDENCE_REF_RE.finditer(vlog):
        used.add(int(m.group(1)))
    return sorted(used)


def _build_system_prompt(
    query: str,
    is_detail: bool,
    intent_category: str | None,
    disease_name: str | None,
    evidence_count: int = 0,
) -> str:
    """Build an adaptive system prompt based on what the user is asking."""

    topic_names: dict[str, str] = {
        "Definisi": "Definisi",
        "Etiologi": "Etiologi dan Faktor Risiko",
        "Patogenesis": "Patogenesis dan Patofisiologi",
        "Anamnesis": "Anamnesis",
        "Manifestasi_Klinis": "Manifestasi Klinis",
        "Pemeriksaan_Fisik": "Pemeriksaan Fisik",
        "Diagnosis": "Diagnosis dan Pemeriksaan Penunjang",
        "Tatalaksana": "Tatalaksana",
        "Komplikasi": "Komplikasi dan Prognosis",
        "Prognosis": "Prognosis",
    }

    if is_detail:
        section_instruction = """Buatlah pembahasan KOMPREHENSIF yang mencakup section-section berikut (jika informasi tersedia di referensi):
1. Definisi
2. Etiologi dan Faktor Risiko
3. Patogenesis dan Patofisiologi
4. Anamnesis / Keluhan
5. Pemeriksaan Fisik
6. Pemeriksaan Penunjang
7. Diagnosis
8. Tatalaksana
9. Komplikasi dan Prognosis

Untuk setiap section, gali informasi SEDALAM mungkin dari referensi. Jika suatu section tidak memiliki data di referensi, JANGAN masukkan section tersebut sama sekali (jangan tulis "tidak tersedia")."""
    elif intent_category:
        topic = topic_names.get(intent_category, intent_category)
        section_instruction = f"""User bertanya SPESIFIK tentang: **{topic}**.

PENTING: Fokuskan jawaban HANYA pada topik "{topic}". Buatlah pembahasan yang MENDALAM dan DETAIL untuk topik tersebut saja.
- JANGAN menambahkan section lain yang tidak ditanyakan.
- Jika ada informasi diagram/bagan/alur di referensi, DESKRIPSIKAN secara naratif dan terstruktur.
- Buat jumlah section sesuai kebutuhan (bisa 1-3 section yang semuanya relevan dengan topik).
- Gunakan sub-bullet, penomoran, atau tabel untuk memperjelas."""
    else:
        section_instruction = """Analisis pertanyaan user dan buat section yang paling relevan.
- Jika pertanyaan umum tentang suatu penyakit, buat ringkasan klinis yang mencakup aspek-aspek utama yang tersedia di referensi.
- Jika pertanyaan spesifik, fokuskan pada topik yang diminta.
- JANGAN masukkan section yang tidak memiliki data di referensi.
- Jumlah section fleksibel: bisa 1 hingga 8 tergantung ketersediaan informasi."""

    return f"""Anda adalah Asisten Klinis (Medical RAG) profesional berbasis referensi terverifikasi.

PRINSIP UTAMA:
1. Jawab HANYA berdasarkan DOKUMEN REFERENSI yang diberikan dalam tag <evidence>.
2. Informasi dari referensi adalah SUMBER UTAMA. AI hanya bertugas menyusun dan memperjelas informasi tersebut agar lebih mudah dipahami.
3. KEMAMPUAN VISION (EKSTRAKSI PROTOKOL): Jika Anda menerima input gambar berupa flowchart/bagan tatalaksana, Anda WAJIB mengubah alur visual tersebut menjadi pedoman prosedural langkah-demi-langkah (IF-THEN-ELSE) secara berurutan.
4. RESOLUSI KONFLIK PEDOMAN: Jika menemukan perbedaan data antar referensi, buat baris "Perbandingan Pedoman".
5. CLINICAL REASONING ENGINE (LEVEL KONSULEN):
   - Anda HARUS menyertakan "Diagnosis Banding (DDx)" berdasarkan kemiripan gejala klinis, JIKA ada dalam referensi.
   - Anda HARUS memunculkan "Red Flags" atau kondisi gawat darurat yang wajib diwaspadai, JIKA disinggung dalam referensi.
6. AGENTIC SELF-REFLECTION: LLM WAJIB melakukan validasi internal sebelum menjawab! Buat "verification_log" di awal hasil JSON-mu, dan pastikan setiap angka dosis dan prosedur benar-benar tertera di evidence. Jangan berhalusinasi dosis!
7. Gunakan bahasa Indonesia formal medis. Tebalkan (**bold**) istilah medis, gunakan _italic_ untuk nama latin.

EVIDENCE COVERAGE CHECK:
- Anda menerima {evidence_count} evidence documents.
- Pastikan SETIAP evidence digunakan minimal 1x jika relevan.
- Di verification_log, sebutkan evidence mana saja yang Anda gunakan dan yang tidak relevan.

SISTEM CITATION GRANULAR:
- Setiap klaim/fakta medis WAJIB diakhiri dengan citation format [E1], [E2], dst. yang merujuk ke <evidence id="N">.
- Setiap kalimat yang menyebut angka, dosis, atau prosedur HARUS memiliki citation.
- Contoh: **Aspirin** diberikan dosis loading **160-320 mg** per oral [E3].
- Anda boleh menggabungkan citation: [E1][E3] atau [E2, E5].

{section_instruction}

FORMAT OUTPUT (RAW JSON VALID, TANPA markdown code block):
{{
  "verification_log": "Catatan verifikasi: Evidence yang digunakan: [E1], [E2], ... Evidence tidak relevan: [EN] karena ...",
  "disease": "Nama Penyakit/Kondisi",
  "sections": [
    {{
      "title": "Judul Section",
      "markdown": "Konten dalam **Markdown** dengan citation [E1] di setiap klaim.\\nGunakan:\\n- **Bold** untuk istilah penting\\n- _Italic_ untuk nama latin\\n- Bullet points untuk daftar\\n- Tabel | untuk data komparatif\\n- Penomoran untuk alur/tahapan"
    }}
  ],
  "citations": ["Sumber p.Halaman"]
}}

ATURAN KETAT:
- Field WAJIB "markdown" (bukan "points")
- Output HARUS berupa raw JSON valid
- Gunakan \\n untuk baris baru di dalam JSON string
- JANGAN masukkan section dengan konten "Tidak tersedia di referensi"
- Gunakan [E1], [E2] dst. untuk SETIAP klaim medis penting
"""


def _build_extraction_prompt(evidence_count: int) -> str:
    """Build system prompt for Pass 1 (fact extraction) of two-pass synthesis."""
    return f"""Anda adalah mesin ekstraksi fakta medis. Tugas Anda HANYA mengekstrak fakta dari {evidence_count} evidence documents yang diberikan.

INSTRUKSI:
1. Baca SEMUA evidence yang diberikan dalam tag <evidence>.
2. Ekstrak SEMUA fakta penting per section klinis.
3. Setiap fakta HARUS disertai referensi [E1], [E2], dst.
4. JANGAN menambahkan informasi yang tidak ada di evidence.
5. JANGAN membuat narasi. Hanya buat bullet list fakta.

FORMAT OUTPUT (RAW JSON VALID):
{{
  "extracted_facts": [
    {{
      "section": "Nama Section (Definisi/Etiologi/Patogenesis/dst)",
      "facts": [
        "Fakta 1 dari evidence [E1]",
        "Fakta 2 dari evidence [E3]"
      ]
    }}
  ],
  "evidence_used": [1, 2, 3],
  "evidence_unused": [4, 5]
}}
"""


def _call_copilot(
    copilot_token: str,
    messages: list[dict[str, Any]],
) -> str:
    """Make a single Copilot API call and return raw content string."""
    body = {
        "messages": messages,
        "model": "gpt-4.1",
        "temperature": 0.1,
        "stream": False,
        "max_tokens": 4096,  # Ensure long answers aren't cut off
    }

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(COPILOT_CHAT_URL, data=data, headers={
        'Authorization': f'Bearer {copilot_token}',
        'Content-Type': 'application/json',
        'User-Agent': 'GitHubCopilotChat/0.1.0',
        'Editor-Version': 'vscode/1.92.0',
        'Editor-Plugin-Version': 'copilot-chat/0.18.0',
        'Openai-Organization': 'github-copilot',
        'Openai-Intent': 'conversation-panel',
    }, method='POST')

    # region agent log
    _agent_debug_log(
        "H2",
        "copilot_request_meta",
        {
            "model": body["model"],
            "payload_bytes": len(data),
            "message_count": len(messages),
            "has_multimodal_user": any(
                isinstance(m.get("content"), list) for m in messages if m.get("role") == "user"
            ),
        },
    )
    # endregion

    try:
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            return resp_data['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')[:2000]
        # region agent log
        _agent_debug_log(
            "H1-H5",
            "copilot_http_error",
            {
                "status": e.code,
                "model": body["model"],
                "payload_bytes": len(data),
                "message_count": len(messages),
                "error_body_preview": err_body,
            },
        )
        # endregion
        raise Exception(f"Copilot API error: {e.code} — {err_body}") from e


def _parse_json_response(raw_content: str) -> dict[str, Any]:
    """Parse LLM response, stripping markdown fences if present."""
    content = raw_content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())


def _attach_images_to_content(
    user_content: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> None:
    """Encode and append base64 images to the user content array."""
    for img in images:
        img_path = img.get("image_abs_path", "")
        if not img_path:
            continue
        try:
            mime_type, _ = mimetypes.guess_type(img_path)
            mime_type = mime_type or "image/jpeg"
            with open(img_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                })
        except Exception as e:
            print(f"[WARN] Failed to encode image for Vision API: {e}")


def _postprocess_response(
    parsed_json: dict[str, Any],
    sorted_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Clean up, resolve citations, compute coverage metadata."""
    # Migrate legacy "points" → "markdown"
    if "sections" in parsed_json:
        for sec in parsed_json["sections"]:
            if "points" in sec and "markdown" not in sec:
                sec["markdown"] = "\n".join(f"- {p}" for p in sec["points"])
                del sec["points"]

        parsed_json["sections"] = [
            sec for sec in parsed_json["sections"]
            if sec.get("markdown", "").strip()
            and "tidak tersedia" not in sec.get("markdown", "").lower()
            and "belum tersedia" not in sec.get("markdown", "").lower()
            and sec.get("markdown", "").strip() != "-"
        ]

    # Collect used evidence IDs before resolving
    used_ids = _extract_used_evidence_ids(parsed_json)

    # Resolve [E1] → (Source, Hal N) in all markdown sections
    for sec in parsed_json.get("sections", []):
        if "markdown" in sec:
            sec["markdown"] = _resolve_evidence_citations(sec["markdown"], sorted_evidence)
    if "verification_log" in parsed_json:
        parsed_json["verification_log"] = _resolve_evidence_citations(
            parsed_json["verification_log"], sorted_evidence
        )

    # Build real citations from evidence
    unique_citations: list[str] = []
    for item in sorted_evidence:
        c = f"{item['source_name']} p.{item['page_no']}"
        if c not in unique_citations:
            unique_citations.append(c)
    parsed_json["citations"] = unique_citations

    # Evidence coverage metadata
    total = len(sorted_evidence)
    unused_ids = sorted(set(range(1, total + 1)) - set(used_ids))
    parsed_json["evidence_coverage"] = {
        "total_evidence": total,
        "used_evidence": used_ids,
        "unused_evidence": unused_ids,
        "coverage_percent": round(len(used_ids) / total * 100, 1) if total else 0,
    }
    parsed_json["grounded"] = True

    return parsed_json


def ask_copilot_adaptive(
    disease_name: str,
    evidence: list[dict[str, Any]],
    github_token: str,
    chat_history: list[dict[str, Any]] | None = None,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call Copilot API with structured evidence, granular citations, and optional two-pass synthesis."""
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")

    copilot_token = get_copilot_token(github_token)

    context_text, sorted_evidence = _format_evidence_structured(evidence)

    intent_category = _extract_topic_intent(disease_name)
    is_detail = _is_detail_request(disease_name)
    detected_disease = _extract_disease_name(disease_name)

    user_prompt = f"Query Klinis: {disease_name}\n\nDokumen Referensi Tersedia:\n{context_text}"

    try:
        if is_detail and len(evidence) > 6:
            # Two-pass synthesis for detail requests with many evidence chunks
            result = _two_pass_synthesis(
                copilot_token, disease_name, user_prompt, context_text,
                sorted_evidence, intent_category, detected_disease,
                chat_history, images,
            )
        else:
            result = _single_pass_synthesis(
                copilot_token, disease_name, user_prompt,
                sorted_evidence, is_detail, intent_category, detected_disease,
                chat_history, images,
            )

        return _postprocess_response(result, sorted_evidence)

    except Exception as e:
        print("Copilot API Error:", e)
        return {
            "disease": disease_name,
            "sections": [{
                "title": "AI Processing Error",
                "markdown": f"Terjadi kesalahan saat memproses: `{e}`",
            }],
            "citations": [
                f"{item['source_name']} p.{item['page_no']}"
                for item in evidence[:5]
            ],
            "grounded": False,
        }


def _single_pass_synthesis(
    copilot_token: str,
    query: str,
    user_prompt: str,
    sorted_evidence: list[dict[str, Any]],
    is_detail: bool,
    intent_category: str | None,
    disease_name: str | None,
    chat_history: list[dict[str, Any]] | None,
    images: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Standard single-pass LLM synthesis."""
    system_prompt = _build_system_prompt(
        query=query,
        is_detail=is_detail,
        intent_category=intent_category,
        disease_name=disease_name,
        evidence_count=len(sorted_evidence),
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    if images:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        _attach_images_to_content(user_content, images)
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    raw = _call_copilot(copilot_token, messages)
    return _parse_json_response(raw)


def _two_pass_synthesis(
    copilot_token: str,
    query: str,
    user_prompt: str,
    context_text: str,
    sorted_evidence: list[dict[str, Any]],
    intent_category: str | None,
    disease_name: str | None,
    chat_history: list[dict[str, Any]] | None,
    images: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Two-pass synthesis: first extract facts, then compose narrative."""
    # Pass 1: Fact extraction
    extraction_prompt = _build_extraction_prompt(len(sorted_evidence))
    pass1_messages: list[dict[str, Any]] = [
        {"role": "system", "content": extraction_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_extraction = _call_copilot(copilot_token, pass1_messages)
    extracted = _parse_json_response(raw_extraction)

    facts_text = ""
    for section_data in extracted.get("extracted_facts", []):
        facts_text += f"\n### {section_data.get('section', 'General')}\n"
        for fact in section_data.get("facts", []):
            facts_text += f"- {fact}\n"

    # Pass 2: Narrative synthesis from extracted facts + original evidence
    system_prompt = _build_system_prompt(
        query=query,
        is_detail=True,
        intent_category=intent_category,
        disease_name=disease_name,
        evidence_count=len(sorted_evidence),
    )

    synthesis_user = (
        f"Query Klinis: {query}\n\n"
        f"FAKTA TEREKSTRAK DARI EVIDENCE (gunakan ini sebagai panduan, tapi tetap rujuk evidence asli):\n"
        f"{facts_text}\n\n"
        f"EVIDENCE ASLI:\n{context_text}"
    )

    pass2_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                pass2_messages.append({"role": role, "content": content})

    if images:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": synthesis_user}]
        _attach_images_to_content(user_content, images)
        pass2_messages.append({"role": "user", "content": user_content})
    else:
        pass2_messages.append({"role": "user", "content": synthesis_user})

    raw_synthesis = _call_copilot(copilot_token, pass2_messages)
    return _parse_json_response(raw_synthesis)


def refine_markdown_with_instruction(
    markdown: str,
    instruction: str,
    github_token: str,
) -> str:
    """Revise library Markdown per user instruction (Copilot). Returns full Markdown body."""
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")
    copilot_token = get_copilot_token(github_token)
    system = """Anda adalah editor medis. Revisi artikel Markdown yang diberikan sesuai instruksi pengguna.
Keluaran HANYA berupa Markdown valid. Gunakan bahasa Indonesia medis formal. Jangan mengarang fakta baru."""
    user = f"""## Artikel saat ini (Markdown)

{markdown}

## Instruksi revisi

{instruction}

Tuliskan ulang artikel lengkap dalam Markdown. Tanpa fence ```."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_copilot(copilot_token, messages)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _build_mindmap_prompt(disease_name: str, competency_level: str | None) -> str:
    """Build a comprehensive system prompt for mindmap generation from a medical article."""
    level_note = f" (Level Kompetensi SKDI: {competency_level})" if competency_level else ""
    return f"""Anda adalah pakar pendidikan kedokteran yang ahli dalam membuat peta konsep (mindmap) visual untuk membantu mahasiswa kedokteran belajar.

TUGAS ANDA:
Baca artikel medis tentang **{disease_name}{level_note}** yang diberikan dan bangun struktur MINDMAP KOMPREHENSIF yang kaya informasi.

PRINSIP MINDMAP:
1. **ROOT NODE** (level 0): Nama penyakit sebagai pusat. Summary berisi definisi singkat 1 kalimat.
2. **SECTION NODES** (level 1): Setiap topik klinis utama = 1 node. Wajib ada jika tersedia:
   - Definisi & Epidemiologi
   - Etiologi & Faktor Risiko
   - Patogenesis & Patofisiologi
   - Anamnesis & Manifestasi Klinis
   - Pemeriksaan Fisik
   - Pemeriksaan Penunjang
   - Diagnosis (Kriteria/Klasifikasi)
   - Tatalaksana (Farmakologi & Non-Farmakologi)
   - Komplikasi
   - Prognosis & Pencegahan
3. **CONCEPT NODES** (level 2): Setiap konsep medis penting dalam tiap section = 1 node. Contoh untuk Tatalaksana: "Lini Pertama", "Dosis Loading", "Monitoring", "Rujukan". BUAT SEBANYAK MUNGKIN konsep.
4. **FACT NODES** (level 3): Fakta spesifik, angka, kriteria, dosis, red flags = 1 node. Contoh: "Aspirin 160 mg PO stat", "Troponin > 0.04 ng/mL", "SpO2 < 90% → oksigen".

ATURAN PENTING:
- SUMMARY setiap node WAJIB berisi informasi substantif (bukan hanya label). Minimum 1 kalimat lengkap.
- JANGAN buat node dengan summary kosong atau hanya "lihat artikel".
- Untuk Tatalaksana: pecah per obat/prosedur, cantumkan dosis jika ada di artikel.
- Untuk Diagnosis: pecah per kriteria diagnostik, nilai cut-off, pemeriksaan spesifik.
- Untuk Red Flags & Kegawatan: buat sebagai fact nodes terpisah yang menonjol.
- ID node harus unik, lowercase, gunakan underscore. Contoh: "ttl_aspirin", "dx_kriteria_framingham".
- JUMLAH NODE: Minimal 25 node, idealnya 40-80 node jika artikel kaya informasi.
- EDGES: Setiap node non-root HARUS memiliki tepat 1 parent edge.

FORMAT OUTPUT (RAW JSON VALID, TANPA markdown code block):
{{
  "disease": "{disease_name}",
  "competency_level": "{competency_level or ''}",
  "summary_root": "Ringkasan 2-3 kalimat tentang penyakit ini secara keseluruhan.",
  "nodes": [
    {{"id": "root", "label": "{disease_name}", "type": "root", "level": 0, "summary": "Definisi singkat 1 kalimat."}},
    {{"id": "definisi", "label": "Definisi & Epidemiologi", "type": "section", "level": 1, "summary": "Ringkasan definisi dan data epidemiologi penting."}},
    {{"id": "def_batasan", "label": "Batasan", "type": "concept", "level": 2, "summary": "Definisi klinis lengkap dari artikel."}},
    {{"id": "def_insidens", "label": "Insidensi", "type": "fact", "level": 3, "summary": "Angka prevalensi/insidensi spesifik jika tersedia."}}
  ],
  "edges": [
    {{"source": "root", "target": "definisi"}},
    {{"source": "definisi", "target": "def_batasan"}},
    {{"source": "def_batasan", "target": "def_insidens"}}
  ],
  "visual_refs": [
    {{"image_url": "", "heading": "Judul gambar/bagan jika disebutkan di artikel", "description": "Deskripsi singkat isi visual"}}
  ],
  "key_takeaways": ["Poin hafalan kunci 1", "Poin hafalan kunci 2", "Poin hafalan kunci 3"]
}}

ATURAN OUTPUT:
- Output HARUS berupa raw JSON valid (bukan markdown, bukan ```json fence).
- Gunakan \\n untuk baris baru di dalam string JSON.
- Semua text dalam Bahasa Indonesia medis formal.
- Tebalkan istilah medis kunci dengan **bold** dalam summary.
"""


def generate_mindmap_from_article(
    disease_name: str,
    markdown_content: str,
    github_token: str,
    competency_level: str | None = None,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive mindmap from a library article using Copilot."""
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")

    copilot_token = get_copilot_token(github_token)
    system_prompt = _build_mindmap_prompt(disease_name, competency_level)

    user_content_text = (
        f"Artikel Medis — {disease_name}:\n\n"
        f"{markdown_content}\n\n"
        "Bangun mindmap komprehensif berdasarkan seluruh isi artikel di atas."
    )

    if images:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_content_text}]
        _attach_images_to_content(user_content, images)
        user_msg: Any = {"role": "user", "content": user_content}
    else:
        user_msg = {"role": "user", "content": user_content_text}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        user_msg,
    ]

    try:
        raw = _call_copilot(copilot_token, messages)
        parsed = _parse_json_response(raw)

        # Ensure required fields exist
        parsed.setdefault("disease", disease_name)
        parsed.setdefault("competency_level", competency_level or "")
        parsed.setdefault("nodes", [])
        parsed.setdefault("edges", [])
        parsed.setdefault("visual_refs", [])
        parsed.setdefault("key_takeaways", [])

        # Inject size/group values for frontend rendering
        type_group = {"root": 0, "section": 1, "concept": 2, "fact": 3}
        type_val = {"root": 20, "section": 12, "concept": 7, "fact": 4}
        for node in parsed["nodes"]:
            node["group"] = type_group.get(node.get("type", "concept"), 2)
            node["val"] = type_val.get(node.get("type", "concept"), 5)

        return parsed

    except Exception as e:
        print(f"[ERROR] Mindmap generation failed for {disease_name}: {e}")
        return {
            "disease": disease_name,
            "competency_level": competency_level or "",
            "nodes": [],
            "edges": [],
            "visual_refs": [],
            "key_takeaways": [],
            "error": str(e),
        }


def merge_two_markdown_articles(
    markdown_base: str,
    markdown_candidate: str,
    github_token: str,
) -> str:
    """
    Gabungkan artikel utama dan kandidat baru menjadi satu Markdown koheren (Copilot).
    Menyatukan isi, mengurangi duplikasi, mempertahankan sitasi; tidak mengarang fakta baru.
    """
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")
    copilot_token = get_copilot_token(github_token)
    system = """Anda adalah editor medis senior. Tugas Anda MENYATUKAN dua versi artikel Markdown tentang topik yang sama menjadi SATU artikel utuh.

ATURAN:
1. Keluaran: satu dokumen Markdown dalam bahasa Indonesia medis formal.
2. Gabungkan isi kedua sumber secara detail; hindari pengulangan paragraf atau bullet yang redundan.
3. Susun dengan struktur jelas (# judul penyakit/kondisi, ## subbagian sesuai materi klinis).
4. Pertahankan referensi/sitasi dari sumber ((Sumber) ..., Hal ...), [En], atau sejenisnya.
5. JANGAN menambahkan fakta, dosis, atau langkah yang tidak tertera di salah satu dokumen.
6. Jika ada perbedaan antar sumber, nyatakan singkat dalam satu kalimat perbandingan bila perlu.
7. Keluaran HANYA Markdown valid, tanpa fence ```."""
    user = f"""## Artikel utama (saat ini)

{markdown_base.strip() or "(belum ada teks; gunakan hanya kandidat)"}

---

## Kandidat baru (regenerate)

{markdown_candidate.strip()}

---

Tulis SATU artikel Markdown lengkap hasil penggabungan kedua bagian di atas."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_copilot(copilot_token, messages)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def ask_copilot_for_pure_list(
    topics_data: dict[str, Any],
    github_token: str,
) -> dict[str, Any]:
    """
    Menggunakan AI untuk menyaring daftar mentah dari DB menjadi daftar 'Murni Penyakit'.
    Fokus: Menghapus heading sampah, simbol, atau judul prosedural.
    """
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing")

    # Siapkan data mentah seminimal mungkin untuk hemat token
    raw_list = ""
    for src in topics_data.get("sources", []):
        raw_list += f"\nSOURCE: {src['source_name']}\n"
        for t in src.get("topics", []):
            raw_list += f"- {t['heading']}\n"

    system_prompt = """Anda adalah ahli rekam medis. Tugas Anda adalah MEMBERSIHKAN daftar topik medis.
ATURAN KETAT:
1. Hanya simpan item yang merupakan NAMA PENYAKIT, KONDISI KLINIS, atau TOPIK MEDIS UTAMA.
2. HAPUS: angka saja, simbol (#, $, dll), judul bab umum (Pendahuluan, Daftar Isi, Lampiran), atau instruksi (misal: '1 Jam Pasca...').
3. JANGAN meringkas daftar. Tampilkan SEMUA yang valid.
4. Output harus JSON valid sesuai format.

FORMAT OUTPUT:
{
  "disease": "Daftar Murni Penyakit & Kondisi Medis",
  "sections": [
    {
      "title": "Nama Sumber",
      "markdown": "1. **Nama Penyakit A**\\n2. **Nama Penyakit B**..."
    }
  ],
  "citations": []
}"""

    copilot_token = get_copilot_token(github_token)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"Saring daftar mentah berikut menjadi murni nama penyakit:\n{raw_list}"}
    ]

    try:
        raw = _call_copilot(copilot_token, messages)
        result = _parse_json_response(raw)
        result["grounded"] = True
        return result
    except Exception as e:
        print(f"[ERROR] ask_copilot_for_pure_list: {e}")
        raise e

def ask_copilot_for_list(
    topics_data: dict[str, Any],
    github_token: str,
    max_topics_per_source: int = 50,
) -> dict[str, Any]:
    """
    Format khusus untuk query enumeratif (daftar penyakit/topik).
    Output: JSON dengan sections berisi numbered list per sumber.
    Fallback ke format manual jika Copilot gagal atau data terlalu besar.
    """
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")

    # Build sources text with limits to prevent context overflow
    sources_text = ""
    for src in topics_data.get("sources", []):
        sources_text += f"\n### {src['source_name']}\n"
        topics = src.get("topics", [])
        for t in topics[:max_topics_per_source]:
            sources_text += f"- {t['heading']}\n"
        if len(topics) > max_topics_per_source:
            sources_text += f"- ... (dan {len(topics) - max_topics_per_source} topik lainnya)\n"

    system_prompt = """Anda adalah asisten medical RAG. Berikan daftar lengkap topik/penyakit yang tersedia dalam knowledge base.

ATURAN:
- Format output: JSON valid TANPA markdown fence
- Kelompokkan berdasarkan sumber
- Buat numbered list yang rapi
- Tambahkan ringkasan singkat di awal

FORMAT OUTPUT:
{
  "disease": "Daftar Topik Knowledge Base",
  "sections": [
    {
      "title": "Nama Sumber",
      "markdown": "Terdapat N topik tersedia:\\n1. **Topik 1**\\n2. **Topik 2**\\n..."
    }
  ],
  "citations": []
}"""

    copilot_token = get_copilot_token(github_token)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Topik tersedia dalam knowledge base:\n{sources_text}"},
    ]

    try:
        raw = _call_copilot(copilot_token, messages)
        result = _parse_json_response(raw)
        result.setdefault("grounded", True)
        return result
    except Exception as e:
        print(f"[WARN] ask_copilot_for_list fallback: {e}")
        # Fallback: format manual tanpa Copilot
        sections = []
        total = 0
        for src in topics_data.get("sources", []):
            count = len(src.get("topics", []))
            total += count
            topics_md = f"Terdapat **{count} topik** tersedia:\n" + "\n".join(
                f"{i+1}. **{t['heading']}**"
                for i, t in enumerate(src.get("topics", []))
            )
            sections.append({"title": src["source_name"], "markdown": topics_md})
        return {
            "disease": f"Daftar Topik Knowledge Base ({total} topik dari {len(sections)} sumber)",
            "sections": sections,
            "citations": [],
            "grounded": True,
        }
