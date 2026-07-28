"""
Pengunduh data historis untuk replay engine.

Dijalankan SEKALI (atau sesekali untuk menyegarkan data). Hasilnya disimpan
sebagai CSV di folder `data/`, dan setelah itu `replay_engine.py` bekerja
sepenuhnya OFFLINE -- bisa diulang ratusan kali tanpa memakai kuota API
sama sekali.

Kenapa perlu:
    Sampai sekarang setiap pertanyaan tentang kualitas sinyal hanya bisa
    dijawab dengan menunggu sinyal live terkumpul -- puluhan trade per
    minggu. Dengan data historis, mesin yang sama bisa dijalankan mundur
    dan menghasilkan ratusan sinyal dalam hitungan menit.

Kuota (paket gratis Twelve Data: 800 request/hari, 8/menit):
    3 pair x 4 timeframe = 12 request. Sekitar 1,5% kuota harian.
    Tetap disarankan mematikan live scanner (`/alerts off`) saat menarik,
    karena scanner sendiri sudah memakai porsi besar kuota harian.

Batas provider yang perlu diketahui:
    - Maksimum 5.000 candle per request.
    - Kedalaman data intraday forex/kripto sekitar 1 tahun ke belakang.
    - Instrumen indeks (NAS100/US30) belum tentu tersedia di paket gratis.
      Skrip ini TIDAK menganggapnya gagal total: pair yang tidak tersedia
      dilaporkan apa adanya lalu dilewati.

Contoh pemakaian:
    python download_history.py
    python download_history.py --pairs XAU/USD --timeframes 15min,1h
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd

BASE_URL = "https://api.twelvedata.com/time_series"
DATA_DIR = Path("data")

# Batas keras provider. Jangan dinaikkan -- request akan ditolak.
MAX_OUTPUT_SIZE = 5000

# Paket gratis: 8 request/menit. Jeda 9 detik memberi margin aman terhadap
# jitter jaringan, sekaligus tetap menyelesaikan 12 request dalam ~2 menit.
SECONDS_BETWEEN_REQUESTS = 9.0

DEFAULT_PAIRS = ["XAU/USD", "NAS100", "EUR/USD"]

# M5 sengaja TIDAK diikutkan secara default. Alasannya bukan selera:
# jendela uji multi-timeframe ditentukan oleh timeframe TERKECIL, dan
# 5.000 candle M5 hanya mencakup ~3,5 minggu -- sementara tanpa M5
# jendelanya menjadi ~2,5 bulan (dibatasi M15). Membuang M5 melipatgandakan
# sampel uji sekitar tiga kali lipat tanpa biaya tambahan.
DEFAULT_TIMEFRAMES = ["15min", "30min", "1h", "4h"]

# Nama simbol alternatif di provider. Beberapa instrumen (terutama indeks)
# memakai kode berbeda; dicoba berurutan sampai ada yang berhasil.
PROVIDER_ALIASES: dict[str, tuple[str, ...]] = {
    "XAU/USD": ("XAU/USD",),
    "XAG/USD": ("XAG/USD",),
    "EUR/USD": ("EUR/USD",),
    "GBP/USD": ("GBP/USD",),
    "USD/JPY": ("USD/JPY",),
    "BTC/USD": ("BTC/USD",),
    "ETH/USD": ("ETH/USD",),
    "NAS100": ("NDX", "IXIC", "NAS100", "US100"),
    "US30": ("DJI", "US30", "DJIA"),
}


def _safe_filename(pair: str, timeframe: str) -> str:
    return f"{pair.replace('/', '')}_{timeframe}.csv"


async def _request_candles(
    session: aiohttp.ClientSession,
    provider_symbol: str,
    timeframe: str,
    api_key: str,
) -> tuple[pd.DataFrame | None, str]:
    """
    Satu request ke Twelve Data.

    Mengembalikan (DataFrame, pesan). DataFrame None berarti gagal, dan
    pesannya menjelaskan kenapa -- dibedakan supaya "simbol tidak tersedia"
    tidak tertukar dengan "kuota habis".
    """
    params = {
        "symbol": provider_symbol,
        "interval": timeframe,
        "outputsize": MAX_OUTPUT_SIZE,
        "format": "JSON",
        "apikey": api_key,
    }

    try:
        async with session.get(BASE_URL, params=params, timeout=60) as response:
            if response.status == 429:
                return None, "KUOTA/RATE LIMIT (HTTP 429)"
            payload = await response.json()
    except Exception as exc:  # noqa: BLE001
        return None, f"gagal jaringan: {exc}"

    if isinstance(payload, dict) and payload.get("status") == "error":
        return None, str(payload.get("message", "error tanpa pesan"))

    values = payload.get("values") if isinstance(payload, dict) else None
    if not values:
        return None, "tidak ada data dikembalikan"

    frame = pd.DataFrame(values)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame[["datetime", "open", "high", "low", "close"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return frame, "ok"


async def download(pairs: list[str], timeframes: list[str]) -> int:
    api_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: environment variable TWELVE_DATA_API_KEY belum diisi.")
        print("Contoh:  export TWELVE_DATA_API_KEY=xxxxx")
        return 1

    DATA_DIR.mkdir(exist_ok=True)

    total_requests = len(pairs) * len(timeframes)
    print(f"Akan menarik {total_requests} request "
          f"({len(pairs)} pair x {len(timeframes)} timeframe).")
    print(f"Perkiraan waktu: ~{total_requests * SECONDS_BETWEEN_REQUESTS / 60:.1f} menit")
    print(f"Kuota terpakai: {total_requests} dari 800/hari\n")

    report: list[tuple[str, str, str]] = []
    request_count = 0

    async with aiohttp.ClientSession() as session:
        for pair in pairs:
            aliases = PROVIDER_ALIASES.get(pair, (pair,))
            for timeframe in timeframes:
                frame = None
                message = "belum dicoba"

                for alias in aliases:
                    if request_count > 0:
                        await asyncio.sleep(SECONDS_BETWEEN_REQUESTS)
                    request_count += 1

                    frame, message = await _request_candles(
                        session, alias, timeframe, api_key
                    )
                    if frame is not None:
                        break
                    # Kalau masalahnya kuota, mencoba alias lain hanya
                    # memperparah -- hentikan seluruh proses.
                    if "429" in message or "KUOTA" in message:
                        print(f"\nBERHENTI: {message}")
                        print("Tunggu kuota pulih, lalu jalankan ulang.")
                        return 2

                if frame is None:
                    print(f"  GAGAL  {pair:10} {timeframe:6} -> {message}")
                    report.append((pair, timeframe, f"GAGAL: {message}"))
                    continue

                path = DATA_DIR / _safe_filename(pair, timeframe)
                frame.to_csv(path, index=False)

                span_days = (
                    frame["datetime"].iloc[-1] - frame["datetime"].iloc[0]
                ).days
                info = (
                    f"{len(frame)} candle | "
                    f"{frame['datetime'].iloc[0].date()} s/d "
                    f"{frame['datetime'].iloc[-1].date()} "
                    f"({span_days} hari)"
                )
                print(f"  OK     {pair:10} {timeframe:6} -> {info}")
                report.append((pair, timeframe, info))

    print("\n" + "=" * 64)
    print("RINGKASAN")
    print("=" * 64)
    for pair, timeframe, info in report:
        print(f"{pair:10} {timeframe:6}  {info}")

    print(f"\nFile tersimpan di: {DATA_DIR.resolve()}")
    print("Langkah berikutnya:  python replay_engine.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unduh candle historis untuk replay engine."
    )
    parser.add_argument(
        "--pairs",
        default=",".join(DEFAULT_PAIRS),
        help="Daftar pair dipisah koma.",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(DEFAULT_TIMEFRAMES),
        help="Daftar timeframe dipisah koma.",
    )
    args = parser.parse_args()

    pairs = [item.strip() for item in args.pairs.split(",") if item.strip()]
    timeframes = [
        item.strip() for item in args.timeframes.split(",") if item.strip()
    ]

    return asyncio.run(download(pairs, timeframes))


if __name__ == "__main__":
    sys.exit(main())
