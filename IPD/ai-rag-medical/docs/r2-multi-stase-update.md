# Pembaruan: Migrasi R2 + Arsitektur Multi-Stase

Dokumen ini merangkum perubahan yang diimplementasikan untuk memindahkan penyimpanan gambar ke Cloudflare R2 dan menjadikan **stase** sebagai dimensi data (bukan hanya struktur folder kode).

**Tanggal referensi:** April 2026

---

## Ringkasan

| Area | Perubahan |
|------|-----------|
| **Supabase** | Kolom `stase_slug` pada `chunks`, `images`, `graph_edges`; RPC pencarian vektor & FTS mendukung filter stase opsional |
| **Cloudflare** | Binding R2 `MEDICAL_IMAGES` ke bucket `coas-medical-images`, variabel `R2_PUBLIC_BASE_URL` |
| **Worker** | Parameter `stase_slug` pada API chat, gambar terkait, dan pipeline library; resolusi URL gambar memakai R2 |
| **Indexer (Python)** | `STASE_MATERI_ROOTS` untuk beberapa stase; `stase_slug` di SQLite lokal |
| **Migrasi** | Script upload ke R2; migrasi Supabase mengirim `stase_slug` |
| **Frontend** | State `activeStase` + pemilih stase; payload `stase_slug` ke `/search_disease_context` |

---

## File yang Ditambah atau Diubah

### Baru

- `migration/04_add_stase_slug.sql` — DDL + indeks + definisi ulang fungsi `search_chunks_vector` dan `search_chunks_fts`
- `migration/05_upload_images_to_r2.py` — upload gambar dari SQLite lokal ke R2 (S3-compatible), update `storage_url` dan `stase_slug` di Supabase

### Diubah

| File | Isi |
|------|-----|
| `worker/wrangler.toml` | `[[r2_buckets]]`, `R2_PUBLIC_BASE_URL` |
| `worker/src/types.ts` | `MEDICAL_IMAGES`, `R2_PUBLIC_BASE_URL` pada `Env` |
| `worker/src/retriever.ts` | `staseSlug?` pada `searchChunks`, `relatedImages`; filter RPC & query; fallback URL R2 |
| `worker/src/index.ts` | Schema `stase_slug` untuk search & gambar; propagasi ke pipeline artikel library |
| `src/medrag/config.py` | `STASE_MATERI_ROOTS`, `MATERI_PAGE_GLOB` (glob lama tetap ada untuk kompatibilitas) |
| `src/medrag/models.py` | Field `stase_slug` pada `SourcePage`, `ChunkRecord`, `ImageRecord` |
| `src/medrag/indexer.py` | Discovery multi-stase; kolom `stase_slug` di skema SQLite + insert |
| `migration/02_migrate_to_supabase.py` | Insert `stase_slug` untuk chunks & images; kompatibilitas SQLite tanpa kolom `stase_slug` |
| `frontend/src/App.tsx` | Pemilih stase (pill) + `stase_slug` di payload query |

---

## Struktur Kunci R2 (objek)

Format yang disarankan:

```
{stase_slug}/{source_clean}/{nama_file_gambar}
```

Contoh: `ipd/Nama_Sumber_Buku/img-12.jpeg`

Script `05_upload_images_to_r2.py` membangun key dari `stase_slug`, `source_name` (dinormalisasi), dan nama file dari `image_ref`.

---

## Langkah Operasional (setelah merge kode)

1. **Supabase** — Jalankan isi `migration/04_add_stase_slug.sql` di SQL Editor (sekali).
2. **Cloudflare R2** — Pastikan bucket `coas-medical-images` ada dan public URL sesuai `R2_PUBLIC_BASE_URL` di `wrangler.toml`.
3. **R2 API** — Tambahkan ke `.env.migration` (untuk script upload): `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID`, opsional `R2_BUCKET_NAME`, `R2_PUBLIC_BASE_URL`.
4. **Upload gambar** — `python migration/05_upload_images_to_r2.py` (opsional `--stase ipd`, `--dry-run`).
5. **Worker** — `wrangler deploy` dari folder `worker/` setelah perubahan konfigurasi.
6. **Stase baru** — Tambah pasangan `(slug, "Folder/Materi")` di `STASE_MATERI_ROOTS`, indeks ulang, jalankan migrasi Supabase + upload R2 untuk stase tersebut.

---

## API (request body)

- **`POST /search_disease_context`** — field opsional `stase_slug` (default `"ipd"`).
- **`POST /get_related_images`** — field opsional `stase_slug` (default `"ipd"`).

---

## Catatan

- Frontend menampilkan pemilih stase (mis. IPD, Saraf, Anak, ObGyn); data RAG untuk stase selain IPD muncul setelah materi diindeks dan dimigrasikan dengan `stase_slug` yang sama.
- Binding R2 pada Worker tersedia untuk ekstensi mendatang (mis. proxy atau validasi objek); URL publik gambar utama tetap lewat kolom `storage_url` di Supabase.
