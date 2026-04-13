from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chunker import extract_context_chunks
from excel_exporter import ExcelMeta, export_quiz_to_xlsx
from generator import generate_quiz_with_metrics
from loaders import load_file


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _run_async(coro):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    return asyncio.run(coro)


def _distribute_total(total: int, enabled: Dict[str, bool]) -> Tuple[int, int, int]:
    order = ["mcq", "tf", "fill"]
    selected = [k for k in order if enabled.get(k)]
    if total <= 0 or not selected:
        return 0, 0, 0
    base, rem = divmod(total, len(selected))
    counts = {k: base for k in selected}
    for k in selected:
        if rem <= 0:
            break
        counts[k] += 1
        rem -= 1
    return counts.get("mcq", 0), counts.get("tf", 0), counts.get("fill", 0)


def _safe_filename(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        cleaned = cleaned.replace(ch, "-" if ch in ("/", "\\", ":", "|") else "")
    return cleaned[:120].strip()


def _build_export_filename(konu: str, difficulty: str) -> str:
    k = _safe_filename(konu)
    d = _safe_filename(str(difficulty))
    if not k:
        return f"{d.lower()} seviye quiz" if d else "quiz"
    return f"{k} {d} seviye quiz" if d else f"{k} quiz"


_QTYPE_LABEL = {
    "mcq": "Çoktan Seçmeli",
    "multiple_choice": "Çoktan Seçmeli",
    "tf": "Doğru / Yanlış",
    "true_false": "Doğru / Yanlış",
    "fill": "Boşluk Doldurma",
    "fill_blank": "Boşluk Doldurma",
    "blank": "Boşluk Doldurma",
}


def _qtype_label(qtype: str) -> str:
    return _QTYPE_LABEL.get((qtype or "").lower().strip(), qtype.upper() or "—")


def _is_type(qtype: str, family: str) -> bool:
    q = (qtype or "").lower().strip()
    if family == "mcq":
        return q in ("mcq", "multiple_choice")
    if family == "tf":
        return q in ("tf", "true_false")
    if family == "fill":
        return q in ("fill", "fill_blank", "blank")
    return False


def _render_question(i: int, q: Dict) -> None:
    qtype = str(q.get("type") or "unknown").lower().strip()
    label = _qtype_label(qtype)

    with st.container(border=True):
        head_left, head_right = st.columns([3, 1])
        with head_left:
            st.markdown(f"**Soru {i}** · {label}")
        with head_right:
            diff = q.get("difficulty")
            if diff is not None:
                st.markdown(
                    f"<div style='text-align:right;color:#888;font-size:12px;'>Zorluk: {diff}/5</div>",
                    unsafe_allow_html=True,
                )

        question_text = str(q.get("question", "")).strip()
        st.markdown(f"#### {question_text}")

        if _is_type(qtype, "mcq"):
            opts = q.get("options") or {}
            correct_key = str(q.get("correct", "")).strip().upper()
            if isinstance(opts, dict) and opts:
                for k in sorted(opts.keys()):
                    is_correct = str(k).strip().upper() == correct_key
                    prefix = "**→**" if is_correct else "   "
                    line = f"{prefix} **{k}.** {opts[k]}"
                    if is_correct:
                        st.success(line)
                    else:
                        st.markdown(line)

        elif _is_type(qtype, "tf"):
            ans = str(q.get("answer") or q.get("correct") or "").strip().lower()
            if ans in ("doğru", "dogru", "true"):
                st.success("Cevap: Doğru")
            else:
                st.error("Cevap: Yanlış")

        elif _is_type(qtype, "fill"):
            ans = str(q.get("answer", "")).strip()
            if ans:
                st.info(f"Cevap: **{ans}**")

        explanation = str(q.get("explanation") or "").strip()
        if explanation:
            st.caption(explanation)

        src = q.get("source") or q.get("context") or ""
        if src:
            with st.expander("Kaynak metin"):
                st.write(src)


st.set_page_config(
    page_title="Quiz Generator Agent",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Quiz Generator Agent")
st.caption("Dokümanınızı yükleyin, Yapay Zeka ile bağlama duyarlı quiz üretin.")


for key, default in {
    "last_quiz": None,
    "last_metrics": {},
    "last_metrics_status": None,
    "last_error_message": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.header("Doküman Yükleme")
    uploaded = st.file_uploader(
        "PDF, PPTX, TXT veya DOCX dosyası yükleyin",
        type=["pdf", "pptx", "txt", "docx"],
        help="Maksimum 200 MB boyutunda dosya yükleyebilirsiniz.",
    )

    st.divider()
    st.header("Ayarlar")

    difficulty = st.selectbox("Zorluk Seviyesi", ["Kolay", "Orta", "Zor"], index=1)
    total_questions = st.number_input(
        "Toplam Soru Sayısı",
        min_value=1, max_value=60, value=10, step=1,
    )

    st.markdown("**Soru Tipleri**")
    mcq_checked = st.checkbox("Çoktan Seçmeli", value=True)
    tf_checked = st.checkbox("Doğru / Yanlış", value=True)
    fill_checked = st.checkbox("Boşluk Doldurma", value=True)

    st.divider()
    st.header("Çıktı Formatı")
    enokta_mode = st.checkbox(
        "Bu quiz enokta'ya yüklenecek",
        value=False,
        help="İşaretlenirse Excel çıktısı Departman / Eğitim / Konu / Amaç / Hazırlayan kolonlarını içerir.",
    )

    departman = egitim = konu = amac = ""
    hazirlayan = "egitim.yonetici"
    if enokta_mode:
        departman = st.text_input("Departman", value="")
        egitim = st.text_input("Eğitim", value="")
        konu = st.text_input("Konu", value="")
        amac = st.text_input("Amaç", value="")
        hazirlayan = st.text_input("Hazırlayan", value="egitim.yonetici")

    st.divider()
    st.markdown(
        "**Akış:**\n\n"
        "1. Dosyayı yükleyin\n"
        "2. Zorluk ve soru tiplerini seçin\n"
        "3. Quiz Oluştur'a basın\n"
        "4. Çıktıyı Excel veya JSON olarak indirin"
    )


mcq_count, tf_count, fill_count = _distribute_total(
    int(total_questions),
    enabled={"mcq": mcq_checked, "tf": tf_checked, "fill": fill_checked},
)
total = mcq_count + tf_count + fill_count


if uploaded is None:
    st.info("Başlamak için kenar çubuğundan bir doküman yükleyin.")
    st.stop()

if not (mcq_checked or tf_checked or fill_checked):
    st.warning("En az bir soru tipi seçmelisiniz.")
    st.stop()

if total <= 0:
    st.warning("Toplam soru sayısı en az 1 olmalı.")
    st.stop()


st.subheader("Quiz Özeti")
sum_cols = st.columns(5)
sum_cols[0].metric("Toplam", total)
sum_cols[1].metric("Çoktan Seçmeli", mcq_count)
sum_cols[2].metric("Doğru / Yanlış", tf_count)
sum_cols[3].metric("Boşluk Doldurma", fill_count)
sum_cols[4].metric("Zorluk", difficulty)

st.caption(
    f"Doküman: **{uploaded.name}** · Çıktı modu: "
    f"**{'enokta uyumlu' if enokta_mode else 'sade'}**"
)

if st.button("Quiz Oluştur", type="primary", use_container_width=True):
    raw = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()
    tmp_path = None

    try:
        with st.spinner("İçerik yükleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

        with st.spinner("Metin çıkarılıyor..."):
            text = load_file(tmp_path)
            preprocessing_metrics: Dict = {}
            paragraphs = extract_context_chunks(text, metrics=preprocessing_metrics)

        if not paragraphs:
            st.session_state.last_quiz = None
            st.session_state.last_metrics = {}
            st.session_state.last_metrics_status = "failed"
            st.session_state.last_error_message = "Dokümandan yeterli içerik çıkarılamadı."
        else:
            with st.spinner("LLM ile sorular üretiliyor... (işlem birkaç dakika sürebilir)"):
                quiz, metrics = _run_async(
                    generate_quiz_with_metrics(
                        paragraphs,
                        mcq_count=mcq_count,
                        tf_count=tf_count,
                        fill_count=fill_count,
                        difficulty=difficulty,
                        preprocessing_metrics=preprocessing_metrics,
                    )
                )
                st.session_state.last_metrics = metrics

            if not quiz:
                st.session_state.last_quiz = None
                st.session_state.last_metrics_status = "failed"
                st.session_state.last_error_message = "Quiz üretilemedi (boş çıktı)."
            elif len(quiz) == 1 and isinstance(quiz[0], dict) and quiz[0].get("type") == "error":
                st.session_state.last_quiz = None
                st.session_state.last_metrics_status = "failed"
                st.session_state.last_error_message = (
                    f"Quiz üretimi hata verdi: {quiz[0].get('error', 'Bilinmeyen hata')}"
                )
            else:
                st.session_state.last_quiz = quiz
                st.session_state.last_metrics_status = "success"
                st.session_state.last_error_message = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if st.session_state.last_metrics_status == "failed":
    st.error(st.session_state.last_error_message or "Bilinmeyen hata")


if st.session_state.last_quiz:
    quiz = st.session_state.last_quiz
    metrics = st.session_state.last_metrics or {}
    skipped = int(metrics.get("skipped_questions", 0))

    if skipped > 0:
        st.success(f"Quiz hazır — Üretilen: **{len(quiz)}** · Atlanan: **{skipped}**")
    else:
        st.success(f"Quiz hazır — Üretilen: **{len(quiz)}**")

    tab_questions, tab_metrics, tab_export = st.tabs([
        "Sorular",
        "Metrikler",
        "Dışa Aktarma",
    ])

    with tab_questions:
        type_counts = {"mcq": 0, "tf": 0, "fill": 0}
        for q in quiz:
            t = str(q.get("type", "")).lower().strip()
            if _is_type(t, "mcq"):
                type_counts["mcq"] += 1
            elif _is_type(t, "tf"):
                type_counts["tf"] += 1
            elif _is_type(t, "fill"):
                type_counts["fill"] += 1

        hc = st.columns(4)
        hc[0].metric("Toplam", len(quiz))
        hc[1].metric("Çoktan Seçmeli", type_counts["mcq"])
        hc[2].metric("Doğru / Yanlış", type_counts["tf"])
        hc[3].metric("Boşluk Doldurma", type_counts["fill"])

        st.divider()

        filter_sel = st.selectbox(
            "Filtrele",
            ["Tümü", "Çoktan Seçmeli", "Doğru / Yanlış", "Boşluk Doldurma"],
            index=0,
        )
        filter_map = {
            "Çoktan Seçmeli": "mcq",
            "Doğru / Yanlış": "tf",
            "Boşluk Doldurma": "fill",
        }

        shown = 0
        for i, q in enumerate(quiz, start=1):
            t = str(q.get("type", "")).lower().strip()
            if filter_sel != "Tümü" and not _is_type(t, filter_map[filter_sel]):
                continue
            _render_question(i, q)
            shown += 1

        if shown == 0:
            st.info("Seçilen filtre için sonuç bulunmuyor.")

    with tab_metrics:
        st.subheader("Üretim Özeti")

        m_coverage_raw = metrics.get("coverage_ratio", 0)
        if isinstance(m_coverage_raw, (int, float)) and m_coverage_raw <= 1:
            m_coverage = f"{int(m_coverage_raw * 100)}%"
        else:
            m_coverage = "—"

        oc = st.columns(4)
        oc[0].metric("Üretilen", len(quiz))
        oc[1].metric("Atlanan", skipped)
        oc[2].metric("İçerik Kapsama", m_coverage)
        oc[3].metric("LLM Çağrısı", int(metrics.get("llm_call_count", 0)))

        st.divider()

        st.markdown("#### Çoktan Seçmeli (Multistage)")
        mc = st.columns(4)
        mc[0].metric("Toplam", int(metrics.get("mcq_total", 0)))
        mc[1].metric("Multistage OK", int(metrics.get("mcq_multistage_success", 0)))
        mc[2].metric("Fallback", int(metrics.get("mcq_fallback_legacy", 0)))
        mc[3].metric("Verify Fail", int(metrics.get("mcq_verify_fail", 0)))

        st.markdown("#### Doğru / Yanlış")
        tc = st.columns(4)
        tc[0].metric("Toplam", int(metrics.get("tf_total", 0)))
        tc[1].metric("Başarılı", int(metrics.get("tf_success", 0)))
        tc[2].metric("Pozitif", int(metrics.get("tf_positive_count", 0)))
        tc[3].metric("Negatif", int(metrics.get("tf_negative_count", 0)))

        st.markdown("#### Boşluk Doldurma")
        fc = st.columns(4)
        fc[0].metric("Toplam", int(metrics.get("fill_total", 0)))
        fc[1].metric("Başarılı", int(metrics.get("fill_success", 0)))
        fc[2].metric("Kalite Fail", int(metrics.get("fill_quality_fail", 0)))
        fc[3].metric("Salvage", int(metrics.get("salvage_triggered_count", 0)))

        st.markdown("#### Zorluk Dağılımı")
        diff_rows = [
            {"Seviye": d, "Adet": int(metrics.get(f"difficulty_count_{d}", 0))}
            for d in range(1, 6)
        ]
        st.dataframe(diff_rows, use_container_width=True, hide_index=True)

        with st.expander("Tüm metrikleri göster"):
            st.json(metrics)

    with tab_export:
        st.subheader("Dosyaya Aktar")

        if enokta_mode:
            st.info(
                "Excel çıktısı **enokta uyumlu** formatta üretilecek "
                "(Departman, Eğitim, Konu, Amaç, Hazırlayan kolonları dahil)."
            )
        else:
            st.info(
                "Excel çıktısı **sade formatta** üretilecek "
                "(sadece soru kolonları, metadata yok)."
            )

        export_base_name = _build_export_filename(konu=konu, difficulty=difficulty)
        meta = ExcelMeta(
            zorluk_derecesi=difficulty,
            departman=departman,
            egitim=egitim,
            konu=konu,
            amac=amac,
            hazirlayan=(hazirlayan or "egitim.yonetici") if enokta_mode else "egitim.yonetici",
        )

        col_xlsx, col_json = st.columns(2)

        try:
            xlsx_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
            export_quiz_to_xlsx(
                quiz=quiz,
                meta=meta,
                out_path=xlsx_path,
                sheet_name="Quiz",
                metrics=metrics,
                include_metadata=enokta_mode,
            )
            with open(xlsx_path, "rb") as f:
                with col_xlsx:
                    st.download_button(
                        "Excel indir",
                        data=f.read(),
                        file_name=f"{export_base_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
        except Exception as e:
            with col_xlsx:
                st.warning(f"Excel export hatası: {e}")

        with col_json:
            st.download_button(
                "JSON indir",
                data=json.dumps(quiz, ensure_ascii=False, indent=2),
                file_name=f"{export_base_name}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()
        st.caption("Excel dosyası ikinci sayfada **Metrics** sayfası ile birlikte gelir.")
