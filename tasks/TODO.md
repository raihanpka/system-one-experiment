# TODO: Eksperimen Column Mapping

Rencana dan keputusan desain: `tasks/PLAN.md`. Kode di `src/`, data di `data/`, hasil di `results/`.

**Model:** hanya `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF` (Q8_0). Tidak ada model lain. Pakai file dan skrip dari repo v3 itu sendiri: runtime v4 menolak GGUF v3, dan sebaliknya.

**Aturan lingkungan (wajib):** semua Python lewat `uv`, tidak pakai `pip`, dan tidak mengunduh Python lain. Pakai Python bawaan: `export UV_PYTHON_DOWNLOADS=never`, `uv venv --python /opt/homebrew/bin/python3`, lalu `uv add ...` dan `uv run ...`. llama.cpp dari brew sudah terpasang (`brew upgrade llama.cpp` hanya jika dibutuhkan).

## Phase 1: Gerbang dan kontrak

- [x] **T0: Spike build `jev-score-v2` di Metal** (selesai, lihat `NOTES.md`)
  - Deskripsi: Unduh seluruh repo (`uvx --from huggingface_hub hf download chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF --local-dir jev-v3-2b-gguf`; unduh satu repo penuh karena card menyebut `jev_score.cpp` padahal filenya `jev_score_v2.cpp`). Clone llama.cpp di commit `441df11f65ea0b6d0c72965aaf70c8241070ddcb` hanya untuk build scorer (binary brew tidak cukup), jalankan `sh build_jev_score.sh <path llama.cpp>` untuk menghasilkan `build/jev-score-v2` (bukan `jev-score`), dan jalankan contoh dari model card dengan `--quant Q8_0`. Dependensi runtime GGUF hanya `tokenizers==0.23.2` dan `numpy==2.5.3`, dipasang dengan `uv pip install -r requirements.txt` di venv uv. Python 3.12 bawaan brew (`/opt/homebrew/opt/python@3.12/libexec/bin/python3`) sama dengan versi yang diuji penulis.
  - Acceptance:
    - [ ] `build/jev-score-v2` ada dan berjalan di M3
    - [ ] Contoh `send/ask_user/discard` menghasilkan `ask_user` dengan probabilitas mendekati 0.860 (F16, Metal, angka di card; Q8_0 boleh sedikit berbeda)
    - [ ] Satu contoh true/false (`noul`) berjalan dan `scores` mentah muncul di output
    - [ ] Waktu per query pendek tercatat
  - Verifikasi: jalankan perintah contoh dan bandingkan dengan model card.
  - Dependencies: None
  - Files: `scripts/setup_jev.sh`, `NOTES.md`
  - Scope: S
  - Catatan: ini gerbang. Isi `build_jev_score.sh` belum dibaca, jadi flag build Metal belum terkonfirmasi. Jika gagal, pindah ke cadangan yang sudah disetujui pengguna, berurutan: (1) MLX (`uv pip install "jev-style[mlx]"`, repo `chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX`, card menyebut paling cepat di Mac), (2) PyTorch MPS (`uv pip install -r requirements.txt` dari repo utama v3, `torch==2.14.0`, `transformers==5.17.0`; cek `python jev_style_decision.py --help` untuk nilai `--device` MPS; sekitar 4x lebih lambat di input panjang). Catat backend yang dipakai di `NOTES.md`; `JevDecider` (T2) menyembunyikan perbedaannya.

- [x] **T1: Kontrak Decisions (meniru OpenAI) dan verifikasi dokumen** (selesai; verifikasi ke halaman resmi OpenAI tertunda karena `ECONNRESET`, lihat `NOTES.md`)
  - Deskripsi: Tipe internal yang bentuknya sama dengan OpenAI Decisions API (`POST /v1/decisions`): pertanyaan `predicate`, `choice` (`choices` dengan `value` dan `description`), `score` (`levels` dengan `label` dan `description`); jawaban `probability`, `choice`/`confidence`/`probabilities`, `score`/`legend`; plus `usage`. Satu interface `Decider.decide(input, questions) -> Response`. Tujuannya: bila nanti pindah ke OpenAI atau Jev hosted, cukup menambah satu adapter. Adapter OpenAI sendiri tidak dibuat sekarang.
  - Acceptance:
    - [ ] `src/decisions/types.py` dan `src/decisions/base.py`
    - [ ] Skema dicocokkan dengan `https://developers.openai.com/api/docs/guides/decisions` (fetch pertama gagal `ECONNRESET`; coba lagi atau pengguna menempelkan isinya) dan `https://docs.typesafe.ai/api/system-one`
    - [ ] Selisih antara dokumen resmi dan sumber pihak ketiga dicatat di `NOTES.md`
    - [ ] Fixture JSON untuk ketiga tipe pertanyaan di `tests/fixtures/`
  - Verifikasi: `uv run python -m pytest tests/test_contract.py` (round-trip fixture tanpa kehilangan field).
  - Dependencies: None
  - Files: `src/decisions/types.py`, `src/decisions/base.py`, `tests/fixtures/*.json`, `tests/test_contract.py`
  - Scope: M

- [x] **T2: `JevDecider`** (selesai)
  - Deskripsi: Implementasi `Decider` yang membungkus `JevStyleDecisionGGUF(quant="Q8_0")` dari repo v3 (metode `decide` dan `score_many`). Pemetaan: predicate → `noul`, choice → `choice`, score → `score`. Skor mentah (`scores`) disimpan di field ekstra `raw_scores`. Mode batch JSONL dan resume.
  - Acceptance:
    - [ ] Respons berbentuk sama dengan fixture T1
    - [ ] `raw_scores` tersedia untuk menghitung ulang temperature secara offline (T global v3 = 0.828; `temperature=1.0` memberi probabilitas tanpa kalibrasi)
    - [ ] Input melebihi 25.600 token (`InputBudgetError`) menghasilkan error yang jelas, bukan crash
  - Verifikasi: test dengan fixture, ditambah contoh model card.
  - Dependencies: T0, T1
  - Files: `src/decisions/jev.py`, `tests/test_jev.py`
  - Scope: M

### Checkpoint: Kontrak (tercapai: Jev jalan, test kontrak lolos)
- [ ] Jev jalan di Mac dan lolos test kontrak
- [ ] Review dengan pengguna

## Phase 2: Data

- [x] **T3: Template kanonik dan aturan transformasi** (selesai: 26 sumber, 31 output, 11 turunan, 6 decoy)
  - Deskripsi: Sekitar 30 kolom output (insurance: polis/klaim; finance: transaksi) dengan tipe, deskripsi, dan aturan transformasi (passthrough, concat, aritmetika, format).
  - Acceptance:
    - [ ] `data/template.json` valid terhadap schema sederhana
    - [ ] Minimal 8 kolom bukan passthrough, minimal 2 kolom dibentuk dari dua input atau lebih
  - Verifikasi: `uv run python -m columnmap.check_template`
  - Dependencies: None
  - Files: `data/template.json`, `src/check_template.py`
  - Scope: S

- [x] **T4: Generator ground truth** (selesai; kasus dibangkitkan dari seed di memori, tidak ada file besar; CSV per kasus lewat `uv run python -m columnmap.dataset.export`; evaluator `src/dsl.py` dikerjakan di sini)
  - Deskripsi: Bangkitkan nama kolom input per tingkat drift 0-6 (0 identik, 1 casing/underscore, 2 singkatan, 3 sinonim, 4 campur ID/EN, 5 absurd, 6 kolom jebakan tanpa padanan), nilai sampel, dan label (input→output, jenis transformasi, ekspresi). Termasuk kasus satu input ke dua output. Untuk tiap output non-passthrough, bangkitkan juga daftar ekspresi kandidat (satu benar, beberapa pengecoh) secara deterministik untuk T6.
  - Acceptance:
    - [ ] Kasus mencakup tujuh tingkat drift, seed tetap, hasil sama di dua kali run
    - [ ] Ekspresi ground truth bisa dieksekusi pada nilai sampel dan menghasilkan nilai output yang benar
    - [ ] Ada kasus 1→2 kolom, kolom jebakan, dan kandidat ekspresi dengan pengecoh
  - Verifikasi: test: dua kali generate menghasilkan file identik; ekspresi menghasilkan nilai yang benar.
  - Dependencies: T3
  - Files: `src/gen_gt.py`, `tests/test_gen_gt.py`, `src/export_csv.py`
  - Scope: M

### Checkpoint: Fondasi (tercapai: 350 kasus, deterministik)
- [ ] Dataset ter-generate dan deterministik
- [ ] Review dengan pengguna

## Phase 3: Eksperimen

- [x] **T5: Runner pemetaan** (selesai; kontrol resume dan bentuk matriks di `tests/test_run_mapping.py`; kondisi konteks sudah di `src/context.py`)
  - Deskripsi: Untuk tiap kasus, satu pertanyaan `predicate` per pasangan (input i, output j), sehingga hasilnya matriks N×M yang independen. Satu pertanyaan `choice` per input untuk jenis transformasi (`passthrough`, `concat`, `arithmetic`, `format`, `lookup`, `none`). Alasan memakai predicate: satu laporan pihak ketiga menyebut probabilitas `choice` sensitif terhadap urutan opsi, sedangkan predicate stabil (belum direplikasi).
  - Acceptance:
    - [ ] `results/jev/*.jsonl` berisi matriks N×M per kasus beserta `raw_scores`
    - [ ] Run bisa dilanjutkan jika terputus
  - Verifikasi: jalankan pada 10 kasus, periksa bentuk matriks dan tidak ada NaN.
  - Dependencies: T2, T4
  - Files: `src/run_mapping.py`, `config.yaml`
  - Scope: M

- [x] **T6: Ranking transformasi kandidat** (selesai; `src/run_transform.py`, evaluator di `src/dsl.py`; smoke 4 run: choice 9-11/11 benar, predicate 6-11/11)
  - Deskripsi: Untuk output hasil transformasi, tanyakan ke Jev (`choice` atas ekspresi kandidat dari T4, dan `predicate` per kandidat). Eksekusi ekspresi yang menang pada nilai sampel dan bandingkan dengan nilai ground truth. Jev tidak menulis ekspresi sendiri, jadi ini menguji apakah ia bisa memilih transformasi yang benar.
  - Acceptance:
    - [ ] Evaluator ekspresi aman (bukan `eval` bebas): kolom, concat, `+ - * /`, format
    - [ ] Metrik: kecocokan pilihan kandidat dan kecocokan nilai hasil, per jenis transformasi
  - Verifikasi: test evaluator (ekspresi valid, invalid, pembagian nol).
  - Dependencies: T5
  - Files: `src/dsl.py`, `src/run_transform.py`, `tests/test_dsl.py`
  - Scope: M

- [x] **T7: Ablasi konteks dan uji urutan opsi** (selesai; `src/context.py`, `src/permute.py`; baris per file 15-30 dengan anomali, sampel 5 nilai merata)
  - Deskripsi: Empat kondisi konteks (nama saja; +deskripsi; +sampel nilai; +keduanya). Untuk pertanyaan `choice`, jalankan beberapa permutasi urutan opsi dan ukur seberapa sering jawaban berubah.
  - Acceptance:
    - [ ] Satu perintah menjalankan semua kondisi
    - [ ] Kondisi dan permutasi tercatat di setiap baris hasil
    - [ ] Persentase jawaban sama antar permutasi dihitung
  - Verifikasi: jalankan pada subset kecil, cek semua kondisi muncul di hasil.
  - Dependencies: T5, T6
  - Files: `src/context.py`, `src/permute.py`
  - Scope: S

- [x] **T8: Metrik, baseline leksikal, dan sweep temperature** (kode selesai dan teruji; `src/metrics.py`; angka sebenarnya muncul setelah run penuh)
  - Deskripsi: Top-1 dan top-3 per tingkat drift dan per konteks; ECE untuk kalibrasi; probabilitas dihitung ulang dari `raw_scores` dengan beberapa nilai T. Baseline pembanding murah tanpa model: kemiripan string (`difflib` dari stdlib), untuk melihat di tingkat drift mana Jev mulai lebih baik.
  - Acceptance:
    - [ ] Satu tabel hasil (CSV dan markdown): metode × drift × konteks
    - [ ] Kurva ECE terhadap T
    - [ ] Kolom jebakan dihitung terpisah (apakah probabilitas tertinggi cukup rendah)
  - Verifikasi: test pada data kecil buatan tangan dengan jawaban yang diketahui.
  - Dependencies: T7
  - Files: `src/metrics.py`, `tests/test_metrics.py`, `results/summary.md`
  - Scope: M

### Checkpoint: Eksperimen
- [ ] Tabel hasil lengkap
- [ ] Semua run bisa diulang dari `config.yaml`
- [ ] Review dengan pengguna

## Phase 4: Laporan

- [ ] **T9: Laporan**
  - Deskripsi: Ringkasan temuan: kurva drift, pengaruh deskripsi dan sampel nilai, pengaruh temperature/kalibrasi, stabilitas urutan opsi, kualitas pemilihan transformasi, kegagalan nyata, dan catatan migrasi ke OpenAI/Jev hosted.
  - Acceptance:
    - [ ] Setiap pertanyaan di `PLAN.md` dijawab dengan angka
    - [ ] Sepuluh contoh kegagalan terburuk dilampirkan
  - Verifikasi: setiap angka bisa ditelusuri ke file di `results/`.
  - Dependencies: T8
  - Files: `results/REPORT.md`
  - Scope: S

### Checkpoint: Selesai
- [ ] Semua pertanyaan di `PLAN.md` terjawab
- [ ] Hasil bisa diulang dari `config.yaml`

## Tambahan atas permintaan pengguna

- [x] Hasil satu proses jadi satu file `result.json` yang rapi (`src/store.py`), bisa dilanjutkan
- [x] Satu perintah: `uv run python src --samples abcd` (`src/pipeline.py`), sampel a sampai g
- [x] Laporan otomatis `results/latest/report/`: ringkasan markdown, grafik SVG Times New Roman (batang berkelompok, batang bertumpuk, garis regresi, heatmap, kalibrasi), CSV prediksi per sampel
- [x] `Makefile` (setup, test, sample, full, pairs, permute, report, data, bench, open, status, stop, clean) dan `README.md` minimalis
- [x] Mode `options` (sekitar 7 kali lebih cepat) dan opsi model di `config.toml` (`quant`, `ngl`, `flash_attn`, `threads`)
- [x] `make sample` dijalankan pada model asli; hasil pertama ada di `NOTES.md` (folder `results/` dikosongkan atas permintaan, dibuat ulang oleh `make sample`)
- [ ] Ukur F16 versus Q8_0 versus Q4_K_M (perlu unduhan) dan pangkas pertanyaan (lihat `NOTES.md`)
- [x] Struktur: slice vertikal (`mapping`, `transform`, `stability`, `convert`, `report`), `core`, `decisions`, `dataset`, `cli` dan `experiment`; tes mengikuti struktur; `make lint`
- [x] Folder run `<samples>_<timestamp>` dengan `result.json`, file hasil konversi, keputusan, matriks probabilitas, `report/summary.md` dan grafik SVG; input, expected, ground truth hanya di `data/samples/`
- [x] README dengan blok bash, contoh `uv run`, dan penjelasan tiap opsi

## Belum dikerjakan

- [ ] Ambang atau skala per field (dari ground truth) supaya field tanpa sumber dan kolom decoy ditolak; bandingkan dengan `--mode pairs`
- [ ] Ukur F16 vs Q8_0 vs Q4_K_M dan pangkas pertanyaan (butuh unduhan, lihat `NOTES.md`)
- [ ] Kondisi konteks tambahan: seluruh baris atau tabel (perubahan prompt saja)
- [ ] Skala lebih besar (lebih banyak kasus per level) untuk interval kepercayaan; saat ini 1 kasus per sampel
- [ ] Verifikasi skema OpenAI Decisions terhadap dokumen resmi (halaman gagal dibuka, `ECONNRESET`) dan adapter OpenAI atau Jev hosted bila diperlukan
- [ ] Ekspor `.xlsx` (perlu `openpyxl`), bila CSV tidak cukup
