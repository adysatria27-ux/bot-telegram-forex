"""
Replay engine -- menguji mesin sinyal terhadap data historis.

PRINSIP UTAMA (jangan dilanggar):
    Replay TIDAK menyalin logika trading. Dia memanggil `get_market_data()`
    yang sama persis dengan yang berjalan di produksi, hanya dengan sumber
    candle diganti ke potongan data historis.

    Kalau replay punya salinan logikanya sendiri, keduanya akan berbeda
    diam-diam dalam hitungan minggu dan hasil backtest menjadi bohong.
    Dengan cara ini, apa pun yang diukur di sini adalah persis apa yang
    akan dilakukan bot.

BIAYA TRANSAKSI:
    Spread dimodelkan. Ini bukan detail kecil: dengan SL 1,5xATR(M15),
    spread ritel memakan sekitar 7% risiko di XAU dan sekitar 12% di
    EUR/USD -- cukup untuk mengubah sistem impas menjadi sistem rugi.
    Semua laporan di bawah SUDAH dikurangi biaya ini.

    Model: satu spread penuh per trade (setengah saat masuk, setengah saat
    keluar), dikonversi ke satuan R lalu dikurangkan dari hasil. Deteksi
    sentuhan SL/TP tetap memakai harga mid, konsisten dengan cara
    signal_tracker menilai trade live.

ATURAN KELUAR:
    Meniru `update_open_signals()` di signal_tracker: SL diperiksa lebih
    dulu pada candle yang sama (konservatif), breakeven setelah TP1, dan
    breakeven dini opsional. Perhitungan R memakai fungsi yang SAMA
    (`_realized_r`) supaya akuntansinya tidak bisa berbeda.

Contoh:
    python replay_engine.py
    python replay_engine.py --pairs XAU/USD --exec-tf 15min
    python replay_engine.py --spread-xau 0.35
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

# Engine memerlukan API key hanya untuk lolos guard di modul analisis.
# Tidak ada satu pun request jaringan yang dilakukan file ini.
os.environ.setdefault("TWELVE_DATA_API_KEY", "REPLAY_OFFLINE")

import xauusd_analysis as xa  # noqa: E402
from signal_tracker import _realized_r, _signal_expiry_hours  # noqa: E402

DATA_DIR = Path("data")

# Spread khas broker ritel, dalam satuan harga instrumen. Ini ESTIMASI --
# ganti dengan angka broker Bapak sendiri lewat argumen agar hasilnya
# mencerminkan kondisi nyata.
DEFAULT_SPREADS: dict[str, float] = {
    "XAU/USD": 0.25,
    "XAG/USD": 0.030,
    "EUR/USD": 0.00008,
    "GBP/USD": 0.00012,
    "USD/JPY": 0.010,
    "BTC/USD": 15.0,
    "ETH/USD": 2.0,
    "NAS100": 1.5,
    "US30": 3.0,
}

# Batas waktu hidup sinyal diambil LANGSUNG dari signal_tracker
# (_signal_expiry_hours), bukan disalin. Sebelumnya nilainya di-hardcode di
# sini, sehingga perubahan aturan expiry di produksi tidak akan pernah
# tercermin di backtest -- persis jenis penyimpangan diam-diam yang harus
# dihindari replay engine.


# =============================================================================
# 1. PEMUATAN DATA
# =============================================================================
def load_frames(pair: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """Memuat CSV hasil download_history.py."""
    frames: dict[str, pd.DataFrame] = {}
    stem = pair.replace("/", "")

    for timeframe in timeframes:
        path = DATA_DIR / f"{stem}_{timeframe}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["datetime"])
        frame = frame.sort_values("datetime").reset_index(drop=True)
        frames[timeframe] = frame

    return frames


# Candle pemanasan yang harus sudah tersedia SEBELUM evaluasi pertama.
# Disamakan dengan OUTPUT_SIZE produksi supaya indikator di replay dihitung
# dari jumlah candle yang sama persis dengan saat bot berjalan live.
WARMUP_CANDLES = 120


def overlapping_window(frames: dict[str, pd.DataFrame]) -> tuple:
    """
    Jendela waktu yang sah untuk backtest multi-timeframe.

    BUKAN sekadar irisan tanggal. Timeframe besar butuh WARMUP_CANDLES
    candle terlebih dahulu sebelum indikatornya bisa dihitung -- kalau
    jendela dimulai dari candle pertama H4, seluruh evaluasi awal akan
    kehilangan H4, coverage jatuh, dan SEMUANYA jadi HOLD karena gate
    kelengkapan data. Hasilnya terlihat seperti "mesin tidak pernah
    memberi sinyal", padahal yang kurang cuma data pemanasan.

    Karena itu awal jendela diambil dari candle ke-WARMUP tiap timeframe,
    bukan candle ke-0.
    """
    starts = []
    for frame in frames.values():
        index = min(WARMUP_CANDLES, len(frame) - 1)
        starts.append(frame["datetime"].iloc[index])

    ends = [frame["datetime"].iloc[-1] for frame in frames.values()]
    return max(starts), min(ends)


# =============================================================================
# 2. SIMULASI KELUAR (meniru signal_tracker.update_open_signals)
# =============================================================================
def simulate_exit(
    signal: str,
    entry: float,
    stop_loss: float,
    targets: tuple[float, float, float],
    future: pd.DataFrame,
    expiry_hours: int,
    breakeven_after_tp1: bool = True,
    breakeven_at_mfe_r: float = 0.0,
) -> dict:
    """
    Menelusuri candle sesudah entry untuk menentukan hasil trade.

    Urutan pemeriksaan sengaja dibuat identik dengan tracker produksi:
    SL diperiksa SEBELUM TP pada candle yang sama, karena dalam satu candle
    kita tidak tahu mana yang tersentuh lebih dulu. Asumsi ini membuat
    laporan sedikit pesimis -- dan itu sehat untuk backtest.
    """
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return {"outcome": "INVALID", "max_tp_hit": 0, "exit_price": None,
                "exit_time": None, "breakeven": False, "mfe_r": 0.0,
                "mae_r": 0.0, "bars": 0}

    direction = 1.0 if signal == "BUY" else -1.0
    tp1, tp2, tp3 = targets

    max_tp_hit = 0
    mfe = 0.0
    mae = 0.0
    deadline = future["datetime"].iloc[0] + pd.Timedelta(hours=expiry_hours)

    for bars, candle in enumerate(future.itertuples(index=False), start=1):
        if candle.datetime > deadline:
            return {
                "outcome": f"EXPIRED_TP{max_tp_hit}" if max_tp_hit else "EXPIRED",
                "max_tp_hit": max_tp_hit,
                "exit_price": float(candle.close),
                "exit_time": candle.datetime,
                "breakeven": False,
                "mfe_r": mfe / risk,
                "mae_r": mae / risk,
                "bars": bars,
            }

        high = float(candle.high)
        low = float(candle.low)

        favorable = (high - entry) if signal == "BUY" else (entry - low)
        adverse = (entry - low) if signal == "BUY" else (high - entry)

        mfe_before = mfe
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        # Breakeven memakai keadaan SEBELUM candle ini diproses, sama
        # seperti tracker -- menghindari asumsi urutan intrabar.
        breakeven_active = (breakeven_after_tp1 and max_tp_hit >= 1) or (
            breakeven_at_mfe_r > 0 and mfe_before >= breakeven_at_mfe_r * risk
        )
        effective_sl = entry if breakeven_active else stop_loss

        sl_hit = low <= effective_sl if signal == "BUY" else high >= effective_sl
        if sl_hit:
            return {
                "outcome": "BREAKEVEN" if breakeven_active else (
                    f"SL_AFTER_TP{max_tp_hit}" if max_tp_hit else "SL"
                ),
                "max_tp_hit": max_tp_hit,
                "exit_price": entry if breakeven_active else stop_loss,
                "exit_time": candle.datetime,
                "breakeven": breakeven_active,
                "mfe_r": mfe / risk,
                "mae_r": mae / risk,
                "bars": bars,
            }

        reached = 0
        for level, price in enumerate((tp1, tp2, tp3), start=1):
            hit = high >= price if signal == "BUY" else low <= price
            if hit:
                reached = level
        max_tp_hit = max(max_tp_hit, reached)

        if max_tp_hit >= 3:
            return {
                "outcome": "TP3", "max_tp_hit": 3, "exit_price": tp3,
                "exit_time": candle.datetime,
                "breakeven": False, "mfe_r": mfe / risk,
                "mae_r": mae / risk, "bars": bars,
            }

    return {
        "outcome": "UNRESOLVED", "max_tp_hit": max_tp_hit,
        "exit_price": float(future["close"].iloc[-1]) if len(future) else None,
        "exit_time": future["datetime"].iloc[-1] if len(future) else None,
        "breakeven": False, "mfe_r": mfe / risk, "mae_r": mae / risk,
        "bars": len(future),
    }


def _forward_returns(
    horizons: tuple[int, ...],
    usable: pd.DataFrame,
    position: int,
    entry: float,
    risk: float,
    direction: float,
) -> dict:
    """
    Pergerakan harga N candle setelah entry, dalam satuan R.

    SENGAJA mengabaikan SL, TP, breakeven, dan expiry. Tujuannya memisahkan
    dua pertanyaan yang selama ini tercampur:

        (a) Apakah ARAH sinyal memprediksi pergerakan harga?
        (b) Apakah GEOMETRI keluar (SL/TP/BE) mengubah prediksi itu
            menjadi uang?

    Semua pengujian sebelumnya hanya mengukur (a) DAN (b) sekaligus, jadi
    kegagalan tidak bisa dilacak ke sumbernya. Kalau (a) nol, tidak ada
    setelan exit apa pun yang bisa menolong -- dan menyetel exit hanya
    membuang waktu.
    """
    out: dict[str, float] = {}
    closes = usable["close"]
    for horizon in horizons:
        target = position + horizon
        if target >= len(closes):
            continue
        move = float(closes.iloc[target]) - entry
        out[f"h{horizon}"] = direction * move / risk
    return out


def _baseline_drift(usable: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    """
    Pembanding: pergerakan harga tanpa syarat apa pun (semua candle).

    Tanpa ini, hasil positif bisa saja cuma memantulkan tren pasar pada
    periode itu, bukan kepintaran sinyalnya. Dinyatakan dalam persen agar
    tidak bergantung pada ukuran risiko tiap trade.
    """
    out: dict[str, float] = {}
    closes = usable["close"].to_numpy()
    for horizon in horizons:
        if horizon >= len(closes):
            continue
        moves = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
        out[f"h{horizon}"] = float(moves.mean() * 100.0)
    return out


def _sweep_variants(
    sweep_values: tuple[float, ...],
    signal: str,
    entry: float,
    stop_loss: float,
    targets: tuple[float, float, float],
    future: pd.DataFrame,
    expiry_hours: int,
    rr_ladder: tuple,
    weights: tuple,
    risk: float,
    spread: float,
) -> dict:
    """
    Menghitung hasil beberapa ambang BREAKEVEN_AT_MFE_R pada trade yang SAMA.

    Ambang breakeven hanya mengubah CARA KELUAR, bukan sinyalnya. Jadi
    seluruh varian bisa dihitung dalam satu kali jalan -- bagian mahal
    (get_market_data) tidak perlu diulang. Karena entry-nya identik,
    perbandingan antar-varian bersifat berpasangan: selisihnya murni
    berasal dari manajemen trade, bukan dari sampel yang berbeda.

    Batasnya: di dunia nyata breakeven membuat trade keluar lebih cepat,
    sehingga slot untuk trade berikutnya terbuka lebih awal. Efek itu TIDAK
    ditangkap di sini. Karena itu pemenang sweep wajib diverifikasi ulang
    lewat satu run penuh dengan --breakeven-mfe-r nilai tersebut.
    """
    variants: dict[str, float] = {}
    for value in sweep_values:
        outcome = simulate_exit(
            signal=signal,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            future=future,
            expiry_hours=expiry_hours,
            breakeven_at_mfe_r=value,
        )
        if outcome["outcome"] in {"INVALID", "UNRESOLVED"}:
            continue
        gross = _realized_r(
            signal=signal,
            entry=entry,
            risk=risk,
            rr_ladder=rr_ladder,
            weights=weights,
            max_tp_hit=outcome["max_tp_hit"],
            exit_price=outcome["exit_price"],
            breakeven=outcome["breakeven"],
        )
        if gross is None:
            continue
        variants[f"be_{value:g}"] = gross - (spread / risk)
    return variants


# =============================================================================
# 3. REPLAY SATU PAIR
# =============================================================================
async def replay_pair(
    pair: str,
    exec_timeframe: str,
    timeframes: list[str],
    atr_timeframe: str,
    spread: float,
    step: int,
    max_steps: int | None,
    breakeven_at_mfe_r: float,
    allow_overlap: bool = False,
    sweep_values: tuple[float, ...] = (),
    horizons: tuple[int, ...] = (),
    invert: bool = False,
) -> dict:
    frames = load_frames(pair, timeframes)
    missing = [tf for tf in timeframes if tf not in frames]
    if missing:
        return {"pair": pair, "error": f"data hilang: {', '.join(missing)}"}
    if exec_timeframe not in frames:
        return {"pair": pair, "error": f"timeframe eksekusi {exec_timeframe} tidak ada"}

    window_start, window_end = overlapping_window(frames)

    # Timeframe yang candle-nya sedikit akan MEMOTONG jendela uji seluruh
    # backtest. Ini harus terlihat di laporan, bukan tersembunyi.
    thin_timeframes = [
        tf for tf, frame in frames.items()
        if len(frame) < WARMUP_CANDLES * 3
    ]

    # --- Menyuntikkan data historis ke mesin produksi ------------------
    # Inilah inti replay: hanya lapisan pengambilan data yang diganti.
    # Seluruh scoring, gate, confidence, dan geometri SL/TP tetap berasal
    # dari kode yang sama dengan yang berjalan di Railway.
    state = {"cutoff": None}

    async def fake_fetch(session, symbol, interval, **kwargs):
        frame = frames.get(interval)
        if frame is None:
            return None
        visible = frame[frame["datetime"] <= state["cutoff"]]
        # Ambang minimum dibiarkan longgar; _score_timeframe punya
        # pemeriksaan jumlah baris sendiri dan akan mengembalikan None
        # dengan pesan yang lebih tepat kalau datanya kurang.
        if len(visible) < 40:
            return None
        return visible.tail(xa.OUTPUT_SIZE).reset_index(drop=True).copy()

    async def fake_resolve(asset):
        return asset.provider_symbols[0]

    original_fetch = xa._fetch_ohlc
    original_resolve = xa._resolve_provider_symbol
    xa._fetch_ohlc = fake_fetch
    xa._resolve_provider_symbol = fake_resolve

    exec_frame = frames[exec_timeframe]
    asset = xa.get_asset_config(pair)
    expiry = _signal_expiry_hours(asset.asset_class, atr_timeframe)
    partial_weights = xa._normalized_partial_weights()

    # Mulai setelah cukup candle terkumpul untuk seluruh indikator.
    usable = exec_frame[
        (exec_frame["datetime"] >= window_start)
        & (exec_frame["datetime"] <= window_end)
    ].reset_index(drop=True)

    trades: list[dict] = []
    holds = 0
    blockers: dict[str, int] = {}
    primary_blockers: dict[str, int] = {}
    evaluated = 0
    errors = 0
    skipped_busy = 0

    # Posisi terbuka terakhir. Selama trade belum keluar, evaluasi
    # berikutnya dilewati -- lihat catatan "SATU POSISI PER PAIR".
    busy_until = None

    indices = range(0, len(usable) - 1, max(1, step))
    if max_steps:
        indices = list(indices)[:max_steps]

    try:
        for position in indices:
            state["cutoff"] = usable["datetime"].iloc[position]

            # SATU POSISI PER PAIR (default).
            # Tanpa aturan ini, satu setup yang bertahan 11 candle akan
            # dicatat sebagai 11 trade berisi gerakan pasar YANG SAMA:
            # Total R membesar sebanding 1/step dan sampel terlihat jauh
            # lebih besar daripada jumlah kejadian independen yang
            # sebenarnya. Produksi juga tidak begitu -- ada cooldown alert
            # dan fingerprint dedup di signal_tracker.
            if (
                not allow_overlap
                and busy_until is not None
                and state["cutoff"] <= busy_until
            ):
                skipped_busy += 1
                continue

            # Cache 30 detik milik produksi harus dibersihkan tiap langkah,
            # kalau tidak seluruh replay hanya menganalisis satu titik waktu.
            xa.clear_caches()

            try:
                analysis = await xa.get_market_data(pair)
            except Exception:  # noqa: BLE001
                errors += 1
                continue

            evaluated += 1
            signal = analysis.get("signal")

            if signal not in {"BUY", "SELL"}:
                holds += 1
                for code in analysis.get("hold_blockers") or []:
                    blockers[code] = blockers.get(code, 0) + 1
                primary = analysis.get("hold_blocker_primary")
                if primary:
                    primary_blockers[primary] = (
                        primary_blockers.get(primary, 0) + 1
                    )
                continue

            entry = float(analysis["entry"])
            stop_loss = float(analysis["sl"])
            targets = (
                float(analysis["tp1"]),
                float(analysis["tp2"]),
                float(analysis["tp3"]),
            )

            if invert:
                # DIAGNOSTIK: ambil sisi berlawanan dari sinyal.
                #
                # Seluruh geometri dicerminkan terhadap harga entry
                # (x -> 2*entry - x). Ini sah karena SL dan TP dihitung
                # simetris dari entry: SL = atr_mult x ATR, TP = RR x risk.
                # Hasilnya jarak risiko dan tangga RR PERSIS sama, cuma
                # arahnya terbalik -- jadi perbandingannya adil.
                #
                # Yang TIDAK ikut terbalik: seluruh gate (arah, sideways,
                # false_breakout) tetap dirancang untuk logika searah tren.
                # Karena itu mode ini hanya alat diagnosa, BUKAN strategi
                # siap pakai. Hasil positif di sini berarti "ada informasi
                # yang dipakai terbalik", bukan "tinggal dibalik saja".
                signal = "SELL" if signal == "BUY" else "BUY"
                stop_loss = (2 * entry) - stop_loss
                targets = tuple((2 * entry) - level for level in targets)

            risk = abs(entry - stop_loss)
            if risk <= 0:
                continue

            future = usable.iloc[position + 1:]
            if future.empty:
                continue

            result = simulate_exit(
                signal=signal,
                entry=entry,
                stop_loss=stop_loss,
                targets=targets,
                future=future,
                expiry_hours=expiry,
                breakeven_at_mfe_r=breakeven_at_mfe_r,
            )
            if result["outcome"] in {"INVALID", "UNRESOLVED"}:
                continue

            gross_r = _realized_r(
                signal=signal,
                entry=entry,
                risk=risk,
                rr_ladder=(
                    analysis["risk_reward_tp1"],
                    analysis["risk_reward_tp2"],
                    analysis["risk_reward_tp3"],
                ),
                weights=partial_weights,
                max_tp_hit=result["max_tp_hit"],
                exit_price=result["exit_price"],
                breakeven=result["breakeven"],
            )
            if gross_r is None:
                continue

            # Biaya spread: satu spread penuh per trade, dikonversi ke R.
            spread_cost_r = spread / risk
            net_r = gross_r - spread_cost_r

            trades.append({
                "time": state["cutoff"],
                "signal": signal,
                "confidence": analysis["confidence_pct"],
                "entry": entry,
                "risk": risk,
                "outcome": result["outcome"],
                "max_tp_hit": result["max_tp_hit"],
                "mfe_r": result["mfe_r"],
                "mae_r": result["mae_r"],
                "bars": result["bars"],
                "gross_r": gross_r,
                "spread_cost_r": spread_cost_r,
                "net_r": net_r,
                "overextension": analysis.get("overextension_ratio_pct"),
                "exit_time": result.get("exit_time"),
                **_forward_returns(
                    horizons=horizons,
                    usable=usable,
                    position=position,
                    entry=entry,
                    risk=risk,
                    direction=1.0 if signal == "BUY" else -1.0,
                ),
                **_sweep_variants(
                    sweep_values=sweep_values,
                    signal=signal,
                    entry=entry,
                    stop_loss=stop_loss,
                    targets=targets,
                    future=future,
                    expiry_hours=expiry,
                    rr_ladder=(
                        analysis["risk_reward_tp1"],
                        analysis["risk_reward_tp2"],
                        analysis["risk_reward_tp3"],
                    ),
                    weights=partial_weights,
                    risk=risk,
                    spread=spread,
                ),
            })

            if not allow_overlap:
                busy_until = result.get("exit_time")
    finally:
        xa._fetch_ohlc = original_fetch
        xa._resolve_provider_symbol = original_resolve

    return {
        "pair": pair,
        "exec_timeframe": exec_timeframe,
        "window": (window_start, window_end),
        "thin_timeframes": thin_timeframes,
        "evaluated": evaluated,
        "errors": errors,
        "skipped_busy": skipped_busy,
        "allow_overlap": allow_overlap,
        "sweep_values": sweep_values,
        "horizons": horizons,
        "invert": invert,
        "baseline_drift": _baseline_drift(usable, horizons),
        "holds": holds,
        "blockers": blockers,
        "primary_blockers": primary_blockers,
        "trades": trades,
        "spread": spread,
    }


# =============================================================================
# 4. LAPORAN
# =============================================================================
def report(result: dict) -> None:
    pair = result["pair"]
    if "error" in result:
        print(f"\n{pair}: DILEWATI -- {result['error']}")
        return

    trades = result["trades"]
    start, end = result["window"]

    print("\n" + "=" * 68)
    print(f"{pair}  |  eksekusi {result['exec_timeframe']}")
    print("=" * 68)
    print(f"Jendela data   : {start.date()} s/d {end.date()}")
    thin = result.get("thin_timeframes")
    if thin:
        print(f"PERINGATAN     : data tipis di {', '.join(thin)} -- "
              "jendela uji dipersempit oleh timeframe ini")
    print(f"Titik evaluasi : {result['evaluated']}"
          + (f"  (gagal: {result['errors']})" if result["errors"] else ""))
    if result.get("allow_overlap"):
        print("Mode posisi    : TUMPANG TINDIH (angka R menggandakan "
              "gerakan yang sama -- hanya untuk perbandingan)")
    else:
        print(f"Mode posisi    : satu posisi per pair "
              f"({result.get('skipped_busy', 0)} evaluasi dilewati "
              "karena posisi masih terbuka)")
    print(f"HOLD           : {result['holds']} "
          f"({result['holds'] / max(result['evaluated'], 1) * 100:.1f}%)")
    print(f"Sinyal         : {len(trades)}")
    print(f"Spread dipakai : {result['spread']}")
    if result.get("invert"):
        print("Mode arah      : DIBALIK (diagnostik) -- BUY mesin "
              "dieksekusi SELL, dan sebaliknya")

    if not trades:
        print("\nTidak ada sinyal pada periode ini.")
        _print_blockers(result)
        return

    frame = pd.DataFrame(trades)
    gross = frame["gross_r"].sum()
    net = frame["net_r"].sum()
    cost = frame["spread_cost_r"].sum()

    print()
    print(f"Total R (kotor)      : {gross:+.2f}R")
    print(f"Biaya spread         : {-cost:+.2f}R "
          f"({cost / len(frame):.3f}R per trade)")
    print(f"Total R (BERSIH)     : {net:+.2f}R")
    print(f"Ekspektasi bersih    : {net / len(frame):+.4f}R per trade")

    wins = int((frame["net_r"] > 0).sum())
    print(f"Trade profit         : {wins}/{len(frame)} "
          f"({wins / len(frame) * 100:.1f}%)")

    print("\nSebaran outcome:")
    for outcome, count in frame["outcome"].value_counts().items():
        subset = frame[frame["outcome"] == outcome]
        print(f"  {outcome:20} {count:5}  "
              f"{subset['net_r'].sum():+8.2f}R")

    print("\nArah:")
    for signal, count in frame["signal"].value_counts().items():
        subset = frame[frame["signal"] == signal]
        print(f"  {signal:5} {count:5} sinyal  {subset['net_r'].sum():+8.2f}R")

    print("\nPer bucket confidence:")
    buckets = pd.cut(
        frame["confidence"],
        bins=[0, 60, 70, 80, 90, 101],
        labels=["<60", "60-69", "70-79", "80-89", "90+"],
        right=False,
    )
    for bucket, group in frame.groupby(buckets, observed=True):
        print(f"  {str(bucket):7} {len(group):5} trade  "
              f"{group['net_r'].sum():+8.2f}R  "
              f"({group['net_r'].mean():+.4f}R/trade)")

    print(f"\nMFE rata-rata: {frame['mfe_r'].mean():.2f}R  |  "
          f"MAE rata-rata: {frame['mae_r'].mean():.2f}R  |  "
          f"durasi rata-rata: {frame['bars'].mean():.0f} candle")

    _print_horizons(result, frame)
    _print_sweep(result, frame)
    _print_blockers(result)


def _print_horizons(result: dict, frame: pd.DataFrame) -> None:
    """Daya prediksi arah, lepas dari mekanika SL/TP."""
    horizons = result.get("horizons") or ()
    columns = [(h, f"h{h}") for h in horizons if f"h{h}" in frame.columns]
    if not columns:
        return

    drift = result.get("baseline_drift") or {}
    print("\nUJI DAYA PREDIKSI ARAH (tanpa SL/TP/breakeven):")
    print(f"  {'Candle':>7} {'n':>5} {'rata2 R':>9} {'t':>7} "
          f"{'menang':>8} {'drift pasar':>12}")
    for horizon, column in columns:
        series = frame[column].dropna()
        if len(series) < 5:
            continue
        mean = series.mean()
        stderr = series.std(ddof=1) / (len(series) ** 0.5)
        tstat = mean / stderr if stderr > 0 else 0.0
        share = (series > 0).mean() * 100
        print(f"  {horizon:>7} {len(series):>5} {mean:+9.4f} {tstat:+7.2f} "
              f"{share:>7.1f}% {drift.get(column, 0.0):>+11.4f}%")
    print("  t di atas +2 = arah sinyal punya daya prediksi nyata.")
    print("  t sekitar 0  = arah sinyal tidak lebih baik dari tebakan.")


def _print_sweep(result: dict, frame: pd.DataFrame) -> None:
    """Perbandingan berpasangan antar ambang breakeven dini."""
    values = result.get("sweep_values") or ()
    columns = [f"be_{value:g}" for value in values]
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return

    print("\nSWEEP BREAKEVEN DINI (entry identik, hanya cara keluar berbeda):")
    print(f"  {'BE at':>8} {'Total R':>10} {'R/trade':>10} "
          f"{'Profit':>9} {'Rugi penuh':>11}")
    baseline = None
    for value, column in zip(values, columns):
        series = frame[column]
        total = series.sum()
        if baseline is None:
            baseline = total
        wins = int((series > 0.001).sum())
        full_losses = int((series < -0.9).sum())
        label = "mati" if value <= 0 else f"{value:g}R"
        print(f"  {label:>8} {total:+10.2f} {series.mean():+10.4f} "
              f"{wins:>6}/{len(series):<3} {full_losses:>11}")
    print("  Catatan: efek 'keluar lebih cepat membuka slot trade baru' "
          "TIDAK dihitung di sini.")
    print("  Verifikasi pemenangnya dengan satu run penuh "
          "--breakeven-mfe-r <nilai>.")


def _print_blockers(result: dict) -> None:
    blockers = result.get("blockers") or {}
    if not blockers:
        return
    primary = result.get("primary_blockers") or {}
    if primary:
        print("\nPENYEBAB UTAMA HOLD (gate pertama yang gagal):")
        for gate, count in sorted(primary.items(), key=lambda i: i[1],
                                  reverse=True):
            share = count / max(result["holds"], 1) * 100
            print(f"  {gate:24} {count:5} ({share:.0f}%)")

    print("\nSemua gate (termasuk yang cuma akibat):")
    ranked = sorted(blockers.items(), key=lambda item: item[1], reverse=True)
    for gate, count in ranked:
        share = count / max(result["holds"], 1) * 100
        print(f"  {gate:24} {count:5} ({share:.0f}%)")


# =============================================================================
# 5. CLI
# =============================================================================
def _timeframe_minutes(timeframe: str) -> int:
    """Mengubah label timeframe menjadi menit untuk pengurutan."""
    text = timeframe.strip().lower()
    if text.endswith("min"):
        return int(text[:-3])
    if text.endswith("h"):
        return int(text[:-1]) * 60
    if text.endswith("day") or text.endswith("d"):
        return int(text.rstrip("day").rstrip("d") or 1) * 1440
    raise ValueError(f"Timeframe tidak dikenal: {timeframe}")


def _timeframe_weights(timeframes: list[str]) -> dict[str, float]:
    """
    Bobot per timeframe untuk daftar APA PUN, bukan cuma yang hardcoded.

    Produksi memakai tangga 5min=1.0 s/d 4h=3.0, yaitu +0.5 tiap naik satu
    anak tangga, dengan timeframe TERBESAR selalu bernilai 3.0. Aturan itu
    dipertahankan di sini: makin besar timeframe, makin berat, jarak antar
    tangga tetap 0.5. Untuk 30min/1h/4h hasilnya persis 2.0/2.5/3.0 --
    identik dengan produksi, jadi konfigurasi lama tidak berubah sedikit pun.

    Diperlukan supaya jendela uji bisa diperpanjang lewat timeframe besar
    (2h, 8h, 1day) tanpa mengubah kode produksi.
    """
    ordered = sorted(timeframes, key=_timeframe_minutes)
    count = len(ordered)
    return {
        tf: round(3.0 - 0.5 * (count - 1 - index), 2)
        for index, tf in enumerate(ordered)
    }



async def run(args) -> int:
    pairs = [item.strip() for item in args.pairs.split(",") if item.strip()]
    timeframes = [
        item.strip() for item in args.timeframes.split(",") if item.strip()
    ]

    # Mesin harus memakai persis daftar timeframe yang datanya tersedia,
    # kalau tidak coverage_ratio akan jatuh dan seluruh sinyal ter-veto
    # oleh gate kelengkapan data.
    xa.TIMEFRAMES = list(timeframes)
    xa.TIMEFRAME_WEIGHTS = _timeframe_weights(timeframes)
    xa.HIGHER_TIMEFRAMES = tuple(
        tf for tf in timeframes if _timeframe_minutes(tf) >= 30
    )
    xa.ENTRY_TIMEFRAME_FOR_ATR = args.atr_tf

    print(f"Timeframe dianalisis : {timeframes}")
    print(f"Bobot                : {xa.TIMEFRAME_WEIGHTS}")
    print(f"ATR untuk SL/TP      : {xa.ENTRY_TIMEFRAME_FOR_ATR}")
    print(f"Tangga RR            : {xa.RR_TARGET_TP1}/{xa.RR_TARGET_TP2}"
          f"/{xa.RR_TARGET_TP3}")
    print(f"Ambang confidence    : {xa.MIN_CONFIDENCE_FOR_SIGNAL}")
    print(f"Bobot partial        : {xa._normalized_partial_weights()}")

    sweep_values = tuple(
        float(item) for item in args.sweep_breakeven.split(",") if item.strip()
    ) if args.sweep_breakeven else ()

    horizons = tuple(
        int(item) for item in args.horizon_test.split(",") if item.strip()
    ) if args.horizon_test else ()

    spreads = dict(DEFAULT_SPREADS)
    if args.spread_xau is not None:
        spreads["XAU/USD"] = args.spread_xau

    summary = []
    for pair in pairs:
        result = await replay_pair(
            pair=pair,
            exec_timeframe=args.exec_tf,
            timeframes=timeframes,
            atr_timeframe=args.atr_tf,
            spread=spreads.get(pair, 0.0),
            step=args.step,
            max_steps=args.max_steps,
            breakeven_at_mfe_r=args.breakeven_mfe_r,
            allow_overlap=args.allow_overlap,
            sweep_values=sweep_values,
            horizons=horizons,
            invert=args.invert,
        )
        report(result)
        if "error" not in result and result["trades"]:
            frame = pd.DataFrame(result["trades"])
            if args.dump_csv:
                stem = pair.replace("/", "")
                path = f"{stem}_{args.dump_csv}"
                frame.to_csv(path, index=False)
                print(f"\nRincian per-trade disimpan: {path}")
            summary.append((pair, len(frame), frame["net_r"].sum(),
                            frame["net_r"].mean()))

    if summary:
        print("\n" + "=" * 68)
        print("RINGKASAN AKHIR (sudah dikurangi spread)")
        print("=" * 68)
        print(f"{'Pair':12} {'Sinyal':>8} {'Total R':>12} {'R/trade':>12}")
        for pair, count, total, mean in summary:
            print(f"{pair:12} {count:8} {total:+12.2f} {mean:+12.4f}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay engine offline.")
    parser.add_argument("--pairs", default="XAU/USD,NAS100,EUR/USD")
    parser.add_argument("--timeframes", default="15min,30min,1h,4h")
    parser.add_argument("--exec-tf", default="15min",
                        help="Timeframe tempat sinyal dievaluasi.")
    parser.add_argument("--atr-tf", default="15min",
                        help="Timeframe sumber ATR untuk SL/TP.")
    parser.add_argument("--step", type=int, default=1,
                        help="Loncat n candle tiap evaluasi (percepat uji).")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--breakeven-mfe-r", type=float, default=0.0,
                        help="Breakeven dini pada n x R (0 = mati).")
    parser.add_argument("--spread-xau", type=float, default=None)
    parser.add_argument(
        "--sweep-breakeven",
        default="",
        help="Uji beberapa ambang breakeven dini sekaligus, mis. "
             "'0,0.4,0.5,0.6,0.75,1.0'. Satu kali jalan, entry identik.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="DIAGNOSTIK: ambil sisi berlawanan dari tiap sinyal, "
             "geometri dicerminkan terhadap entry. Bukan strategi siap pakai.",
    )
    parser.add_argument(
        "--horizon-test",
        default="",
        help="Ukur pergerakan harga N candle setelah sinyal, mis. "
             "'4,12,24,48'. Mengabaikan SL/TP -- menguji arahnya saja.",
    )
    parser.add_argument(
        "--dump-csv",
        default="",
        help="Simpan rincian per-trade, mis. --dump-csv trades.csv",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Izinkan trade tumpang tindih (perilaku lama, "
             "menggandakan hasil -- hanya untuk pembanding).",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print("Folder data/ belum ada. Jalankan dulu:")
        print("  python download_history.py")
        return 1

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
