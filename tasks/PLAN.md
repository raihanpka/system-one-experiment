# Implementation Plan: Eksperimen Column Mapping dengan Decision Model Lokal

## Overview

Eksperimen untuk mengukur seberapa andal model lokal memetakan kolom input berformat bebas ke kolom kanonik template (finance/insurance), termasuk kasus absurd, kolom yang butuh transformasi (concat, hitung), dan skor probabilitas per kolom target. Model tunggal: Jev-Style-2B-Decision-v3 (GGUF, Q8_0) di Mac M3 16GB. Ground truth dibangkitkan sintetis dengan tingkat drift bertahap.

Tugas dan checklist ada di `tasks/TODO.md`.

## Pertanyaan yang dijawab

1. Bisakah satu kolom input menghasilkan probabilitas ke semua kolom output, dan argmax-nya benar?
2. Seberapa cepat akurasi turun saat penamaan menjauh (tingkat drift 0-6)?
3. Apa pengaruh deskripsi atau sampel nilai kolom dibanding nama saja?
4. Bagaimana pengaruh temperature terhadap kalibrasi probabilitas?
5. Bisakah model mengenali jenis transformasi (passthrough, concat, aritmetika), dan memilih ekspresi yang benar dari kandidat?

## Model yang dipakai: Jev-Style-2B-Decision-v3

- Fine-tune Qwen3.5-2B (Apache-2.0). Hanya memberi probabilitas untuk opsi yang diberikan; tidak menghasilkan teks. Ia tidak bisa menulis ekspresi sendiri.
- Tiga tipe pertanyaan: `choice`, `noul` (benar/salah), `score` (2-10 level).
- Skor opsi = `logit(" yes") - logit(" no")`; probabilitas = `softmax(scores / T)`, T global 0.828. `scores` mentah dikembalikan, jadi temperature bisa dihitung ulang offline.
- Batas konteks 25.600 token. Tanpa batas jumlah opsi yang tertulis (fixture uji memakai sampai 151).
- Perlu runtime penulis (`jev_style_decision_gguf.py` dan `build/jev-score-v2`, di-build terhadap llama.cpp commit `441df11f65ea0b6d0c72965aaf70c8241070ddcb`). llama-server, Ollama, dan LM Studio tidak bisa menghasilkan skornya.
- Q8_0 (2,01 GB): 99,7% top-1 sama dengan FP32. Q4_K_M lebih noisy pada input panjang.
- Hasil di card (dijalankan penulis sendiri): JevBench 73,6%, bagian Hard 47,7%; fin_topic 20 kelas zero-shot 61,1% (ECE 0,065). Jawaban sama saat urutan opsi dibalik hanya 88%, jadi sensitivitas urutan nyata di v3.
- Kenapa v3 dan bukan v4: pilihan pengguna. Scorer `jev-score-v2` dilaporkan sudah dicek di Metal untuk v3; untuk v4 baru di CUDA. Runtime v3 dan v4 saling menolak file satu sama lain.
- Catatan lisensi: kode dan bobot Apache-2.0, tetapi sebagian data latih punya syarat sendiri. Untuk eksperimen ini tidak masalah.

## Architecture Decisions

- **Jev memakai `predicate` (`noul`) per pasangan (input i, output j), bukan `choice`.** `choice` memakai softmax sehingga total probabilitas 1, padahal satu input bisa mengisi dua output. Predicate menghasilkan matriks N×M yang independen. Sensitivitas urutan pada `choice` (88% konsisten) menguatkan pilihan ini.
- **Jenis transformasi ditanyakan lewat `choice`** dengan opsi tetap: `passthrough`, `concat`, `arithmetic`, `format`, `lookup`, `none`.
- **Jev tidak menulis ekspresi.** Pengujian transformasi dilakukan dengan memilih dari kandidat: generator membuat satu ekspresi benar dan beberapa pengecoh, Jev memilih, ekspresi pemenang dieksekusi pada nilai sampel dan dibandingkan dengan nilai ground truth. Tidak butuh model lain.
- **Skor mentah selalu disimpan.** Sweep temperature dihitung ulang offline dari `scores`, tanpa menjalankan model lagi. Pada Jev, temperature hanyalah skalar kalibrasi (tidak mengubah jawaban teratas), jadi yang diukur adalah ECE, bukan keragaman jawaban.
- **Ground truth sintetis dengan seed tetap.** Nama input dibuat per tingkat drift: 0 identik, 1 casing/underscore, 2 singkatan, 3 sinonim, 4 campur ID/EN, 5 absurd, 6 kolom jebakan tanpa padanan. Variasi dibangkitkan dari lebih dari satu sumber (aturan manual dan daftar sinonim) agar tidak bias ke satu gaya.
- **Empat kondisi konteks:** nama saja; nama + deskripsi; nama + sampel nilai; nama + deskripsi + sampel nilai.
- **Baseline pembanding tanpa model:** kemiripan string (`difflib` stdlib), untuk melihat di tingkat drift mana Jev mulai lebih baik.
- **Kontrak meniru OpenAI Decisions API (`POST /v1/decisions`).** Satu interface `Decider.decide(input, questions) -> Response` dengan tipe `predicate`, `choice`, `score`, dan bentuk jawaban seperti OpenAI. Hanya `JevDecider` yang dibuat sekarang. Bila nanti pindah ke OpenAI atau Jev hosted (`/v1/systemone`: `noul` = predicate, `criteria` = `choices`/`levels`, jawaban datar di `answers`), cukup menambah satu adapter. Adapter OpenAI tidak dibuat sekarang.
- **Skema OpenAI belum terverifikasi dari sumber resmi.** Halaman resmi gagal dibuka (`ECONNRESET`). Skema dirangkum dari LocalAI, LangChain, dan berita (Oct 2026: public beta, model `gpt-6-luna`, input $0,10 per juta token, output gratis). T1 memverifikasi ulang.
- **Tooling:** Python bawaan (`/opt/homebrew/bin/python3`, 3.12) dikelola dengan `uv`. Tidak memakai `pip`, dan `UV_PYTHON_DOWNLOADS=never` agar uv tidak mengunduh Python lain. llama.cpp dari brew sudah terpasang; source llama.cpp di commit pin hanya dipakai untuk build `jev-score-v2`.

## Task List

Lihat `tasks/TODO.md` untuk rincian, kriteria penerimaan, dan verifikasi per tugas.

### Phase 1: Gerbang dan kontrak
- T0 Spike: build dan jalankan `jev-score-v2` di Metal
- T1 Kontrak Decisions (meniru OpenAI) dan verifikasi dokumen
- T2 `JevDecider`

### Checkpoint: Kontrak

### Phase 2: Data
- T3 Template kanonik dan aturan transformasi
- T4 Generator ground truth

### Checkpoint: Fondasi

### Phase 3: Eksperimen
- T5 Runner pemetaan
- T6 Ranking transformasi kandidat
- T7 Ablasi konteks dan uji urutan opsi
- T8 Metrik, baseline leksikal, dan sweep temperature

### Checkpoint: Eksperimen

### Phase 4: Laporan
- T9 Laporan

### Checkpoint: Selesai

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `jev-score-v2` gagal build atau salah hitung di Metal (flag build belum dibaca) | High | T0 dikerjakan pertama dan dibandingkan dengan contoh card. Jika gagal, pindah ke MLX lalu PyTorch MPS (sudah disetujui pengguna). `JevDecider` dibuat di balik interface, jadi pergantian backend tidak mengubah tugas lain. |
| Jev tidak mengenal jargon finance/insurance, akurasi rendah (v3 hanya 47,7% di bagian Hard) | Med | Itu hasil eksperimen. Laporkan per tingkat drift dan bandingkan dengan baseline `difflib`. |
| Probabilitas predicate antar kolom tidak sebanding | Med | Evaluasi dengan ranking (top-k) selain ambang probabilitas. Coba normalisasi per input. |
| Sensitivitas urutan opsi pada `choice` | Med | Pemetaan memakai predicate. T7 mengukur konsistensi antar permutasi. |
| Ground truth sintetis bias ke gaya penamaan kita | Med | Variasi dari beberapa sumber. Tambahkan sedikit sampel tulisan tangan sebagai sanity check. |
| Skema OpenAI berbeda dari yang dirangkum | Low | Kontrak terpisah di `src/decisions/types.py`; test kontrak memakai fixture. |
| Q8_0 sedikit berbeda dari F16 pada input panjang | Low | Input di sini pendek (nama kolom dan sampel nilai). Bisa diuji ulang dengan F16 bila hasil mencurigakan. |

## Open Questions

- Diputuskan (pengguna setuju): jika build GGUF/Metal gagal di T0, boleh beralih ke MLX (`jev-style[mlx]`, repo `Jev-Style-2B-Decision-v3-MLX`) atau PyTorch MPS (`jev_style_decision.py`, nilai `--device` untuk MPS belum dikonfirmasi, cek `--help`). Urutan: GGUF/Metal, lalu MLX, lalu MPS. Semua lewat `uv`.
- Jumlah kolom template dan jumlah kasus per tingkat drift. Default awal: sekitar 30 kolom, 50 kasus per tingkat drift.
- Perlu sampel data nyata untuk sanity check? Default: tidak, ditambah nanti jika tersedia.
