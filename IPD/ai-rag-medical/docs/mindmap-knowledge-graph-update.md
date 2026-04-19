# Pembaruan: Knowledge Graph Mind Map (Layout, Fullscreen, UI)

Dokumen ini merangkum perubahan pada panel **Knowledge Graph** di frontend (`frontend/src/App.tsx`) untuk tata letak mind map yang lebih teratur, perilaku zoom, dan stabilitas setelah mode layar penuh.

**Tanggal referensi:** April 2026

---

## Ringkasan

| Area | Perubahan |
|------|-----------|
| **Layout tree** | Posisi node dihitung dengan `d3-hierarchy` (`hierarchy` + `tree`) — algoritma Reingold–Tilford; node memakai `fx`/`fy` tetap; simulasi force dinonaktifkan (`cooldownTicks=0`, dll.) |
| **Rendering node** | Mode LOD (Level of Detail): ketika zoom rendah (`globalScale < 0.65`), kotak kompak; teks tetap dirender dengan ukuran font dalam koordinat graf agar tidak hilang atau tumpang tindih berlebihan |
| **Zoom awal / fit** | `onEngineStop` memanggil `zoomToFit` dengan padding agar graf selaras dengan ukuran container |
| **Ukuran canvas** | `ResizeObserver` + `useLayoutEffect` mengukur `containerRef`; guard `offsetWidth/Height > 0` |
| **Fullscreen** | Tombol layar penuh; `isGraphFullscreen`; `fullscreenchange` hanya sinkronkan state (tanpa remount paksa `graphKey` yang memicu race) |
| **Layout pasca-keluar fullscreen** | `setDimensions` memicu `zoomToFit` setelah ukuran berubah; canvas container `position: absolute; inset: 0` agar tidak mendorong flex layout (header/tombol tidak hilang) |
| **Header toolbar** | Judul dan tombol (Visual, Layar penuh, Edit, Regenerate) dipisah ke dua baris agar tidak tertutup saat panel menyempit |
| **Panel** | `overflow-hidden` dihapus dari wrapper panel kanan (`graphPanelRef`) agar tidak memotong konten; `overflow-hidden` tetap pada area canvas |

---

## Detail teknis

### Tree layout (`computeTreeLayout`)

- Konstanta jarak: `TREE_NODE_W`, `TREE_NODE_H` (horizontal/vertical antar slot).
- Root ditentukan dari `type === 'root'`, atau node tanpa parent, atau node pertama.
- Edge yang membentuk siklus dicegah dengan `visited` saat membangun `TreeDatum`.

### ForceGraph2D

- `width` / `height` dari state `dimensions` yang diukur dari container.
- Canvas wrapper: `absolute inset-0` di dalam parent `flex-1 min-h-0 min-w-0 relative` agar tinggi/lebar mengikuti parent, bukan sebaliknya.

### Fullscreen

- `requestFullscreen()` / `exitFullscreen()` pada elemen `graphPanelRef`.
- Setelah keluar fullscreen, `isGraphFullscreen` berubah → effect ukuran ulang menjalankan pengukuran dan `zoomToFit` saat dimensi berubah.

---

## Dependensi frontend

- `d3-hierarchy` dan `@types/d3-hierarchy` (untuk `hierarchy`, `tree`).

---

## File utama yang terdampak

| File | Peran |
|------|--------|
| `frontend/src/App.tsx` | Panel Knowledge Graph: `computeTreeLayout`, `drawMindmapNode`, `ForceGraph2D`, header, fullscreen, dimensi, instrumentation debug (jika masih ada) |
| `frontend/package.json` | Dependensi `d3-hierarchy` |

---

## Catatan pengujian manual

1. Buka mind map untuk penyakit dengan banyak node (mis. Influenza).
2. Uji zoom in/out — node tidak boleh tumpang tindih berlebihan; teks tetap terbaca di LOD sesuai threshold.
3. Masuk **Layar penuh**, lalu keluar — pastikan tombol toolbar (Visual, Layar penuh, Edit, Regenerate) dan judul tetap terlihat; graf tidak “terpotong” satu sisi karena canvas memaksa layout.

---

## Riwayat singkat isu yang diperbaiki

- **Overlap node:** diganti dari layout DAG/force ke tree deterministik + LOD.
- **Zoom out:** teks di mode LOD dipertahankan dengan font dalam koordinat graf.
- **Pasca-fullscreen:** ukuran canvas/stale dimensi dan flex push; diperbaiki dengan ukuran ulang, `zoomToFit` pada resize, canvas absolut, header dua baris, dan menghapus `overflow-hidden` dari panel induk yang memotong toolbar.
