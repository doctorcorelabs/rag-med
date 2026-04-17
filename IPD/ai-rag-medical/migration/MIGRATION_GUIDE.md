# Medical RAG — Migration Execution Guide

## Files yang Dibuat

```
IPD/ai-rag-medical/
├── migration/
│   ├── 01_supabase_schema.sql      ← Run ini di Supabase SQL Editor
│   ├── 02_migrate_to_supabase.py   ← Jalankan lokal sekali untuk copy data
│   └── .env.migration.example      ← Template credentials
├── worker/                         ← Cloudflare Workers backend (TypeScript)
│   ├── src/
│   │   ├── index.ts                ← Router utama (pengganti api.py)
│   │   ├── retriever.ts            ← RAG search (pengganti retriever.py)
│   │   ├── copilot-client.ts       ← Copilot AI (pengganti copilot_client.py)
│   │   ├── medical-vocab.ts        ← Medical NLP constants + functions
│   │   └── types.ts                ← TypeScript type definitions
│   ├── package.json
│   ├── tsconfig.json
│   └── wrangler.toml
├── netlify.toml                    ← Netlify deploy config
└── frontend/
    ├── .env.development            ← VITE_API_URL=http://127.0.0.1:8010
    ├── .env.production             ← VITE_API_URL=https://...workers.dev
    └── src/App.tsx                 ← Sudah diupdate (import.meta.env.VITE_API_URL)
```

---

## FASE 1 — Setup Supabase (Langkah Manual)

### 1.1 Buat Project Supabase
1. Buka https://supabase.com → Sign up / Login
2. Klik **"New Project"** → Isi nama project → Pilih region terdekat (Singapore)
3. Catat:
   - **Project URL**: `https://xxxxxx.supabase.co`
   - **anon public key** (untuk client)
   - **service_role key** (untuk migration script & backend)

### 1.2 Jalankan SQL Schema
1. Di dashboard Supabase → **SQL Editor** → **"New Query"**
2. Copy-paste isi `migration/01_supabase_schema.sql`
3. Klik **"Run"**
4. Verifikasi tabel sudah terbuat di **Table Editor**

### 1.3 Jalankan Migration Script
```powershell
# Di terminal lokal (bukan Worker):
cd e:\Coas\IPD\ai-rag-medical

# Install dependencies
pip install supabase sentence-transformers tqdm python-dotenv

# Buat file credentials
copy migration\.env.migration.example migration\.env.migration
# Edit .env.migration isi SUPABASE_URL dan SUPABASE_SERVICE_KEY

# Jalankan migration (estimasi 10-30 menit tergantung jumlah data)
python migration/02_migrate_to_supabase.py
```

---

## FASE 2 — Deploy Cloudflare Workers

### 2.1 Install Wrangler dan Login
```powershell
# Install wrangler CLI
npm install -g wrangler

# Login ke Cloudflare (akan buka browser)
wrangler login
```

### 2.2 Set Worker Secrets
```powershell
cd e:\Coas\IPD\ai-rag-medical\worker

wrangler secret put SUPABASE_URL
# → Paste: https://xxxxxx.supabase.co

wrangler secret put SUPABASE_SERVICE_KEY
# → Paste: eyJh... (service_role key)

wrangler secret put GITHUB_TOKEN
# → Paste: ghp_... (GitHub Personal Access Token)
```

### 2.3 Enable Cloudflare Workers AI
1. Di **Cloudflare Dashboard** → Workers & Pages → Settings → AI
2. Aktifkan **Workers AI** (gratis tier tersedia)
3. Pastikan `wrangler.toml` sudah ada binding `[ai]`

### 2.4 Deploy Worker
```powershell
cd e:\Coas\IPD\ai-rag-medical\worker
npm install
npx wrangler deploy

# Output:
# Published medrag-worker (2.5 sec)
# https://medrag-worker.YOUR-SUBDOMAIN.workers.dev
```

### 2.5 Uji Worker
```powershell
curl https://medrag-worker.YOUR-SUBDOMAIN.workers.dev/health
# {"status":"ok","version":"3.0.0","runtime":"cloudflare-workers"}

curl -X POST https://medrag-worker.YOUR-SUBDOMAIN.workers.dev/search_disease_context \
  -H "Content-Type: application/json" \
  -d '{"disease_name":"tuberkulosis"}'
```

---

## FASE 3 — Deploy Frontend ke Netlify

### 3.1 Update .env.production
```
# file: frontend/.env.production
VITE_API_URL=https://medrag-worker.YOUR-SUBDOMAIN.workers.dev
```

### 3.2 Deploy via Netlify CLI
```powershell
npm install -g netlify-cli
netlify login

cd e:\Coas\IPD\ai-rag-medical
netlify deploy --dir=frontend/dist --prod
# Atau gunakan Netlify GUI: drag-drop folder frontend/dist
```

### 3.3 Atau Deploy via GitHub (Recommended)
1. Push code ke GitHub
2. Di Netlify → **"New site from Git"** → Pilih repo
3. Settings:
   - **Base directory**: `IPD/ai-rag-medical/frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
4. Di **Environment variables**: tambahkan `VITE_API_URL=https://medrag-worker.....workers.dev`
5. Klik **Deploy**

---

## Checklist Sebelum Deploy Production

- [ ] Supabase schema sudah dibuat (`01_supabase_schema.sql`)
- [ ] Data sudah dimigrasikan ke Supabase (`02_migrate_to_supabase.py`)
- [ ] Cloudflare Worker secrets sudah di-set
- [ ] Worker AI binding aktif di Cloudflare
- [ ] `frontend/.env.production` diisi URL Worker yang tepat
- [ ] `wrangler.toml` nama worker sesuai dengan yang diinginkan
- [ ] Test `/health` endpoint Worker berhasil
- [ ] Test `/search_disease_context` dengan satu query
- [ ] Frontend di-build ulang dengan env production
- [ ] Netlify deploy berhasil dan path routing berfungsi

---

## Troubleshooting

### "Failed to get Copilot token"
→ Pastikan `GITHUB_TOKEN` secret sudah di-set di Worker dan token masih valid

### "Supabase connection failed"
→ Pastikan `SUPABASE_URL` menggunakan format `https://xxx.supabase.co` (tanpa trailing slash)
→ Gunakan **service_role** key bukan **anon** key

### "search_chunks_vector function not found"
→ Jalankan ulang `01_supabase_schema.sql` di Supabase SQL Editor

### Worker AI embedding kosong
→ Pastikan `[ai]` binding ada di `wrangler.toml` dan Workers AI sudah diaktifkan di dashboard
