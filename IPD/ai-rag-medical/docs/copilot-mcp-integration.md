# Integrasi Copilot Chat dengan Medical RAG API

Dokumen ini menjembatani endpoint API lokal ke tools yang dipanggil custom agent Copilot Chat.

## Tujuan
- Menyediakan dua tool untuk chat:
  - search_disease_context
  - get_related_images
- Tool memanggil API lokal yang berjalan di port 8010.

## Kontrak endpoint yang sudah tersedia
- POST /search_disease_context
- POST /get_related_images

## Mekanisme integrasi yang disarankan
1. Jalankan API lokal dari proyek ini.
2. Buat MCP server kecil sebagai adapter:
   - tool search_disease_context -> HTTP POST ke /search_disease_context
   - tool get_related_images -> HTTP POST ke /get_related_images
3. Daftarkan MCP server tersebut pada konfigurasi Copilot Chat lokal.
4. Gunakan agent medical-rag agar alur jawaban selalu retrieval-first.

## Payload contoh
### search_disease_context
```json
{
  "disease_name": "influenza",
  "detail_level": "detail",
  "top_k": 8,
  "include_images": true
}
```

### get_related_images
```json
{
  "disease_name": "influenza",
  "limit": 3
}
```

## Catatan produksi
- Ubah image_abs_path ke URL static yang bisa diakses klien chat.
- Tambahkan API key/allowlist jika server dibuka lintas jaringan.
- Tambahkan logging request dan response ringkas untuk audit citation.
