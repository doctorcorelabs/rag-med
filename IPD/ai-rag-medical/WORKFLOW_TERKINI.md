# Workflow Medical RAG Terkini

Dokumen ini merangkum alur kerja terbaru sistem Medical RAG setelah patch production Worker terakhir.

## 1. Arsitektur Aktif

Sistem berjalan dengan pendekatan hybrid:

- Local development: Python FastAPI + SQLite + local indexing.
- Production deploy: Cloudflare Worker (Hono.js) + Supabase (PostgreSQL/Vector) + R2 Storage.

Production Worker yang aktif saat ini:

- URL: https://medrag-worker.daivanfebrijuansetiya.workers.dev
- Deploy target: `IPD/ai-rag-medical/worker`

## 2. Alur Query Detail Penyakit

Contoh: "jelaskan secara detail Sindrom Mendelson".

1. Request masuk ke endpoint `POST /search_disease_context`.
2. Sistem mendeteksi mode retrieval `relevant` atau `exhaustive`.
3. Disease resolver mencoba beberapa lapisan:
   - exact match dari medical vocab
   - synonym match
   - dynamic lexicon dari data chunk
   - embedding-based fallback bila tidak ada match eksplisit
4. Query detail dinormalisasi agar kata instruksi seperti "jelaskan", "detail", "lengkap" tidak mengganggu retrieval.
5. Retrieval hybrid dijalankan:
   - FTS/BM25 untuk keyword match
   - vector search untuk semantic match
6. Hasil dari FTS dan vector digabung dengan RRF, lalu didedup dan dipruning.
7. Jika hasil awal lemah atau disease confidence rendah, sistem menjalankan auto-retry retrieval dengan query reformulasi yang lebih kaya.
8. Jika evidence cukup, jawaban disintesis oleh Copilot menjadi output Markdown terstruktur.
9. Jika evidence minim, sistem tetap menjawab konservatif dengan warning kualitas evidence.

## 3. Alur Daftar Penyakit / List Intent

Contoh: "daftar penyakit kardiovaskular dari Atria".

1. Sistem mendeteksi list intent dari kata kunci katalog/daftar.
2. Worker melewati vector search yang berat.
3. Query ke database dilakukan langsung untuk mengambil topik dari `chunks`.
4. Filter sumber diterapkan jika user menyebut sumber tertentu seperti Atria, Mediko, atau Kaplan.
5. Hasil mentah dibersihkan oleh AI pure-list filtering.
6. Jika AI refinement gagal, sistem memakai fallback deterministic agar daftar tetap bersih dan lengkap.

## 4. Evidence Quality dan Guardrails

Sistem kini tidak memaksa AI menjawab detail bila bukti terlalu sedikit.

- Jika evidence count rendah, `evidence_quality` diresponse diset ke `low`.
- Jika evidence memadai, `evidence_quality` diset ke `ok`.
- Untuk evidence minim, jawaban tetap diberikan tetapi dengan catatan keterbatasan.
- Sitasi [E1], [E2], dst. dinormalisasi supaya token out-of-range tidak tampil mentah.

## 5. Confidence Metadata

Jawaban detail kini membawa metadata tambahan untuk observabilitas dan UI.

- `answer_confidence`: skor confidence keseluruhan jawaban.
- `section_confidence_map`: confidence per section.
- `evidence_coverage`: ringkasan evidence yang dipakai dan yang tidak dipakai.
- `retrieval_passes`: jumlah pass retrieval yang dijalankan.

## 6. Output Detail yang Diharapkan

Untuk pertanyaan detail penyakit, output ideal berisi section berikut bila evidence tersedia:

- Definisi
- Etiologi dan faktor risiko
- Patogenesis
- Anamnesis / keluhan
- Pemeriksaan fisik
- Pemeriksaan penunjang
- Diagnosis
- Tatalaksana
- Komplikasi dan prognosis

Jika evidence cukup banyak, sistem akan memakai two-pass synthesis. Jika evidence sedang atau sedikit, sistem tetap menjaga output tetap terstruktur tanpa mengarang isi.

## 7. Deploy and Validation Flow

Setelah perubahan logika di Worker:

1. Jalankan type-check.
2. Deploy Worker dengan `npm run deploy`.
3. Validasi endpoint dengan query detail dan list intent.
4. Jika hasil produksi sudah sesuai, update dokumentasi workflow ini bila ada perubahan alur.

## 8. Catatan Implementasi Terkini

- Disease detection sudah diperluas untuk penyakit yang tidak ada di vocab statis.
- Retrieval memiliki auto-retry untuk kasus out-of-vocab atau confidence rendah.
- Response now lebih transparan karena menyertakan confidence dan evidence quality.
- List intent lebih aman karena tidak bergantung pada hardcode page number.

---

*Updated on 2026-04-20*
