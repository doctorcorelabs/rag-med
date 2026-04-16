# Medical RAG for Markdown Materi

MVP ini mengindeks materi markdown dan gambar dari folder IPD, lalu menyediakan endpoint pencarian konteks penyakit yang siap diintegrasikan ke Copilot Chat melalui MCP/tooling.

## Fitur v0.1
- Index markdown per halaman dari pola: IPD/Materi/**/pages/page-*/markdown.md
- Ekstraksi gambar markdown (alt text, path, relasi halaman)
- SQLite FTS5 untuk retrieval cepat
- Endpoint API:
  - POST /search_disease_context
  - POST /get_related_images

## Struktur
- src/medrag/indexer.py: parser markdown, chunking, index builder
- src/medrag/retriever.py: retrieval hybrid sederhana berbasis FTS + heuristik section
- src/medrag/api.py: API layer FastAPI
- scripts/build_index.py: build database index
- scripts/run_api.py: jalankan API server

## Setup
1. Buat environment Python dan install dependency.
2. Jalankan dari folder proyek ini.
3. Entry point scripts sudah menambahkan `src` ke import path, jadi bisa dijalankan juga dari `E:\Coas` memakai path absolut.

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/build_index.py --workspace-root ../..
python scripts/run_api.py --host 127.0.0.1 --port 8010
python scripts/run_mcp.py
```

## Start Otomatis (Backend + Frontend)

Untuk start development server tanpa bentrok port, gunakan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_dev.ps1
```

Opsi hanya backend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_dev.ps1 -BackendOnly
```

## Contoh request
```powershell
curl -X POST http://127.0.0.1:8010/search_disease_context ^
  -H "Content-Type: application/json" ^
  -d "{\"disease_name\":\"influenza\",\"detail_level\":\"detail\",\"top_k\":8,\"include_images\":true}"
```

## Catatan gambar
- Output API mengembalikan image_abs_path agar UI/chat client bisa menampilkan gambar sebagai link/lampiran.
- Output API dan MCP juga menambahkan image_url supaya gambar bisa ditampilkan dari browser/static route.
- Untuk produksi, disarankan map image_abs_path ke URL static server agar render lebih stabil di berbagai client.

## Panduan Pakai
Lihat [docs/panduan-penggunaan.md](docs/panduan-penggunaan.md) untuk urutan build index, run API, run MCP, dan koneksi ke Copilot Chat.
