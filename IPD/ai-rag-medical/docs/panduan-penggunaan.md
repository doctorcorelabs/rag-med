# Panduan Penggunaan Medical RAG

Panduan ini menjelaskan cara menyalakan index, menjalankan API, lalu menghubungkan tools ke Copilot Chat.

## 1. Persiapan
Masuk ke folder proyek:

```powershell
Set-Location "e:/Coas/IPD/ai-rag-medical"
```

Kalau mau dari folder lain seperti `E:\Coas`, itu juga bisa karena script sudah bootstrap path `src` secara otomatis.

Pastikan dependency sudah terpasang:

```powershell
pip install -r requirements.txt
```

## 2. Build index
Index membaca seluruh markdown per halaman dan gambar yang ada di folder materi.

```powershell
$env:PYTHONPATH = "src"
e:/Coas/.venv/Scripts/python.exe scripts/build_index.py --workspace-root ../..
```

Output sukses akan menampilkan jumlah source_pages, chunks, dan images.

## 3. Jalankan API lokal
API ini dipakai oleh UI, debugging, dan bisa juga dipakai sebagai lapisan di atas MCP.

```powershell
$env:PYTHONPATH = "src"
e:/Coas/.venv/Scripts/python.exe scripts/run_api.py --host 127.0.0.1 --port 8010
```

Cek health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
```

## 4. Coba query manual
Contoh request penjelasan penyakit:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8010/search_disease_context -ContentType "application/json" -Body '{"disease_name":"influenza","detail_level":"detail","top_k":8,"include_images":true}'
```

Contoh request gambar terkait:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8010/get_related_images -ContentType "application/json" -Body '{"disease_name":"influenza","limit":3}'
```

## 5. Jalankan MCP server
MCP server dipakai oleh Copilot Chat agar tool search bisa dipanggil otomatis.

```powershell
$env:PYTHONPATH = "src"
e:/Coas/.venv/Scripts/python.exe scripts/run_mcp.py
```

Jika ingin membuat browser-friendly image URL, jalankan juga API lokal karena MCP server mengembalikan `image_url` yang merujuk ke static route `/materials/...`.

## 6. Hubungkan ke Copilot Chat
1. Buka workspace `E:\Coas` di VS Code.
2. VS Code akan membaca konfigurasi MCP dari [.vscode/mcp.json](../../.vscode/mcp.json).
3. Buka Copilot Chat dan pilih tools dari server `medical-rag`.
4. Jika diminta, klik start/enable pada server MCP yang muncul di UI.
5. Setelah server aktif, gunakan agent `medical-rag` atau prompt penyakit yang sudah dibuat.

## 7. Pola penggunaan di Copilot Chat
Ketik permintaan seperti:
- Jelaskan influenza berdasarkan materi
- Buat ringkasan pertussis dengan gambar terkait
- Bandingkan laringitis akut dan kronis

Agent akan:
1. Mencari evidence dulu.
2. Menyusun jawaban terstruktur.
3. Menampilkan sitasi dan gambar jika tersedia.

## 8. Cara menampilkan gambar
Gambar akan dikembalikan sebagai `image_url`.
- Pada MCP server, gambar dikembalikan sebagai `file:///...` supaya bisa dibuka langsung tanpa perlu API.
- Pada API HTTP, gambar tetap bisa dikembalikan sebagai URL static bila UI ingin menampilkan dari server lokal.

## 9. Troubleshooting
- Jika index kosong, pastikan `--workspace-root` sudah benar.
- Jika gambar tidak muncul, cek apakah file gambar ada di folder page dan API sedang berjalan.
- Jika Copilot Chat tidak melihat tool, pastikan MCP server sudah didaftarkan ulang setelah restart VS Code.
