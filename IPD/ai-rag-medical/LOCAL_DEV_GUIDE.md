# 🛠️ Panduan Pengembangan Lokal (Medical RAG)

Dokumentasi ini menjelaskan cara menjalankan dan mengelola index Medical RAG di mesin lokal Anda.

## 🚀 Cara Menjalankan Aplikasi

Aplikasi ini menggunakan satu script hub utama yaitu `scripts/start_dev.ps1` yang mengotomatisasi backend (FastAPI), frontend (Vite), dan proses indexing.

### 1. Penggunaan Standar (Run & Play)
Gunakan ini jika Anda hanya ingin menjalankan aplikasi tanpa mengubah data materi:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dev.ps1
```
*   **Backend:** Berjalan di `http://127.0.0.1:8010`
*   **Frontend:** Berjalan di `http://127.0.0.1:5173`
*   **Browser:** Otomatis terbuka ke UI web.

### 2. Update Materi Baru (Rebuild Index)
Gunakan ini jika Anda menambah folder stase baru, menambah PDF baru, atau mengedit isi dokumen di folder `<Stase>/Materi`:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dev.ps1 -RebuildIndex
```
*   **Auto-Discovery:** Script otomatis mendeteksi folder seperti `IPD/Materi`, `Saraf/Materi`, dll.
*   **Full Processing:** Melakukan ekstraksi teks, chunking, dan pembuatan vector embedding (memakan waktu beberapa menit).

### 3. Update Cepat (Skip Vector)
Gunakan ini jika Anda hanya memindah folder atau mengubah nama file, namun isi teksnya tidak berubah banyak:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dev.ps1 -RebuildIndex -SkipVector
```
*   Hanya memperbarui database metadata (SQLite) tanpa menghitung ulang embedding vector. Sangat cepat.

---

## ⌨️ Pintasan Visual Studio Code (Recommended)

Untuk kenyamanan, gunakan fitur **Tasks** di VS Code (`Ctrl + Shift + P` -> `Tasks: Run Task`):

| Task Name | Fungsi |
| :--- | :--- |
| `🚀 Start Dev (Quick)` | Menjalankan aplikasi (Skenario #1) |
| `🔄 Refresh Local Index` | Rebuild materi baru secara lokal |
| `☁️ Publish to Cloud` | Sinkronisasi data lokal ke Supabase & Cloudflare R2 |

---

## ⚙️ Konfigurasi Parameter (Opsional)

Anda bisa menambahkan flag tambahan saat menjalankan script:
- `-BackendOnly`: Tidak menjalankan frontend Vite.
- `-NoBrowser`: Tidak otomatis membuka browser saat startup.
- `-ApiPort <Port>`: Mengubah port API (default: 8010).
- `-WebPort <Port>`: Mengubah port Frontend (default: 5173).

---

## ⚠️ Catatan Penting
- Pastikan virtual environment Python (`.venv`) aktif atau path-nya benar di baris 19 `scripts/start_dev.ps1`.
- Database lokal disimpan di file `local_rag_database_v2.db` dan folder `chroma_db/`. Jangan hapus manual kecuali Anda ingin melakukan _clean start_ (full rebuild).
