"""
Medical RAG AI Client — v2.0 (Adaptive Copilot Integration)

Key improvements:
- Dynamic system prompt that adapts to user intent (specific vs general)
- Only generates sections relevant to the question
- Deeply extracts information from referenced pages
- Better JSON parsing and error handling
"""

import json
import urllib.request
import urllib.error
import base64
import mimetypes
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


def _build_system_prompt(
    query: str,
    is_detail: bool,
    intent_category: str | None,
    disease_name: str | None,
) -> str:
    """Build an adaptive system prompt based on what the user is asking."""

    # Map intent to human-readable topic name
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
        # User wants comprehensive coverage
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
        # General question — let AI decide
        section_instruction = """Analisis pertanyaan user dan buat section yang paling relevan.
- Jika pertanyaan umum tentang suatu penyakit, buat ringkasan klinis yang mencakup aspek-aspek utama yang tersedia di referensi.
- Jika pertanyaan spesifik, fokuskan pada topik yang diminta.
- JANGAN masukkan section yang tidak memiliki data di referensi.
- Jumlah section fleksibel: bisa 1 hingga 8 tergantung ketersediaan informasi."""

    return f"""Anda adalah Asisten Klinis (Medical RAG) profesional berbasis referensi terverifikasi.

PRINSIP UTAMA:
1. Jawab HANYA berdasarkan DOKUMEN REFERENSI yang diberikan. JANGAN mengarang atau menambahkan informasi dari pengetahuan umum.
2. Informasi dari referensi adalah SUMBER UTAMA. AI hanya bertugas menyusun dan memperjelas informasi tersebut agar lebih mudah dipahami.
3. KEMAMPUAN VISION (EKSTRAKSI PROTOKOL): Jika Anda menerima input gambar berupa flowchart/algoritma/bagan tatalaksana, Anda WAJIB mengubah alur visual tersebut menjadi pedoman prosedural langkah-demi-langkah (IF-THEN-ELSE). Buat sub-bagian tersendiri seperti "Protokol Visual Tatalaksana" yang membedah cabang-cabang keputusan klinis di dalam gambar ke dalam bentuk teks / list terstruktur.
4. RESOLUSI KONFLIK PEDOMAN: Jika Anda menemukan perbedaan data/pedoman antar sumber referensi yang diberikan (misalnya beda dosis antara buku Atria vs Mediko), JANGAN menggabungkannya secara ambigu. Anda WAJIB membuat sub-bagian "Perbandingan Pedoman" yang secara eksplisit memisahkan apa yang dikatakan oleh masing-masing referensi.
5. Gunakan bahasa Indonesia formal medis. Tebalkan (**bold**) istilah medis penting, gunakan _italic_ untuk nama latin/organisme.

{section_instruction}

FORMAT OUTPUT (RAW JSON VALID, TANPA markdown code block):
{{
  "disease": "Nama Penyakit/Kondisi",
  "sections": [
    {{
      "title": "Judul Section",
      "markdown": "Konten dalam **Markdown**. Gunakan:\\n- **Bold** untuk istilah penting\\n- _Italic_ untuk nama latin\\n- Bullet points untuk daftar\\n- Tabel | untuk data komparatif\\n- Penomoran untuk alur/tahapan"
    }}
  ],
  "citations": ["Sumber p.Halaman"]
}}

ATURAN KETAT:
- Field WAJIB "markdown" (bukan "points")
- Output HARUS berupa raw JSON valid
- Gunakan \\n untuk baris baru di dalam JSON string
- JANGAN masukkan section dengan konten "Tidak tersedia di referensi"
- Setiap poin HARUS menyertakan referensi inline seperti: (Nama_Sumber p.XX)
"""


def ask_copilot_adaptive(
    disease_name: str,
    evidence: list[dict[str, Any]],
    github_token: str,
    chat_history: list[dict[str, Any]] | None = None,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call Copilot API with adaptive prompt, multi-turn history, and Vision API support."""
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing or empty")

    copilot_token = get_copilot_token(github_token)

    # Build rich context from evidence
    context_text = ""
    for item in evidence:
        context_text += (
            f"\n--- Source: {item['source_name']} (Page {item['page_no']}) ---\n"
            f"Heading: {item['heading']}\n"
            f"Section Category: {item.get('section_category', 'General')}\n"
            f"{item['content']}\n"
        )

    # Detect user intent for adaptive prompt
    intent_category = _extract_topic_intent(disease_name)
    is_detail = _is_detail_request(disease_name)
    detected_disease = _extract_disease_name(disease_name)

    system_prompt = _build_system_prompt(
        query=disease_name,
        is_detail=is_detail,
        intent_category=intent_category,
        disease_name=detected_disease,
    )

    user_prompt = f"Query Klinis: {disease_name}\n\nDokumen Referensi Tersedia:\n{context_text}"

    # Build multi-turn messages array
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Insert previous chat history for multi-turn context
    if chat_history:
        for turn in chat_history[-6:]:  # Keep last 3 exchanges
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Prepare user message content (text + images if available)
    if images:
        user_content = [{"type": "text", "text": user_prompt}]
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
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_img}"
                        }
                    })
            except Exception as e:
                print(f"[WARN] Failed to encode image for Vision API: {e}")

        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    body = {
        "messages": messages,
        "model": "gpt-4.1",  # User requested gpt-4.1 for vision
        "temperature": 0.1,
        "stream": False,
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

    try:
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            raw_content = resp_data['choices'][0]['message']['content']

            # Sanitize JSON in case model outputs markdown JSON blocks
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            parsed_json = json.loads(raw_content.strip())

            # Migrate legacy "points" format to "markdown" if needed
            if "sections" in parsed_json:
                for sec in parsed_json["sections"]:
                    if "points" in sec and "markdown" not in sec:
                        sec["markdown"] = "\n".join(f"- {p}" for p in sec["points"])
                        del sec["points"]

                # Remove empty/placeholder sections
                parsed_json["sections"] = [
                    sec for sec in parsed_json["sections"]
                    if sec.get("markdown", "").strip()
                    and "tidak tersedia" not in sec.get("markdown", "").lower()
                    and "belum tersedia" not in sec.get("markdown", "").lower()
                    and sec.get("markdown", "").strip() != "-"
                ]

            # Override citations with real source names from evidence
            real_citations = [
                f"{item['source_name']} p.{item['page_no']}"
                for item in evidence
            ]
            unique_citations: list[str] = []
            for c in real_citations:
                if c not in unique_citations:
                    unique_citations.append(c)

            parsed_json['citations'] = unique_citations
            parsed_json['grounded'] = True
            return parsed_json

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
