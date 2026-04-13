from __future__ import annotations

import json


_QUALITY_TEMPLATE = """
GENEL KURALLAR:
- Sorular ezbere dayalı olmamalıdır.
- Tanım, amaç, sonuç, istisna veya senaryo bilgisi ölçmelidir.
- Aynı kavramı tekrar eden sorular üretme.
- Cümleler açık, tek anlamlı ve eğitim seviyesine uygun olmalıdır.
- Daha önce sorulmuş sorularla aynı/çok benzer sorular üretme.

SORU ÇEŞİTLİLİĞİ:
- Eğer metin uygunsa aşağıdaki türler arasında çeşitlilik sağla:
    - tanım sorusu
    - amaç/sonuç sorusu
    - istisna veya yanlış çıkarım sorusu
    - karşılaştırma sorusu
    - süreç/adım sorusu
- Daha önce üretilen sorularla aynı kavramı veya aynı cümleyi tekrar sorgulama.

ZORLUK:
- Zorluk seviyesi 1-5 arası düşünülmelidir.
- Seviye 1-2: temel kavram bilgisi
- Seviye 3: yorumlama ve ilişkilendirme
- Seviye 4-5: senaryo, istisna veya yanlış çıkarım analizi

ZORLUK HEDEFİ:
Bu sorular yaklaşık {difficulty}/5 zorluk seviyesinde olmalıdır.
Sorular temel/orta/ileri düzey bilişsel becerileri ölçecek şekilde kurgulanmalıdır.

DİL HEDEFİ:
- Çıktı dili yalnızca Türkçe olmalıdır.
- Soru, seçenekler ve açıklama tamamen Türkçe olmalıdır.
- İngilizce soru üretme veya İngilizce cümle kurma.
- İngilizce üretilirse bu soru veya cevap geçersiz sayılır.

NOT:
- Zorluk seviyesini metnin içinde yazma.
- Yalnızca sorunun içeriğini zorluk seviyesine uygun kurgula.
"""


def _quality_block(difficulty: int = 3) -> str:
    try:
        d = max(1, min(5, int(difficulty)))
    except Exception:
        d = 3
    return _QUALITY_TEMPLATE.format(difficulty=d)


def prompt_mcq(sentence: str, difficulty: int = 3) -> str:
    return f"""
Aşağıdaki metne dayanarak 1 adet çoktan seçmeli soru üret.

Metin:
\"\"\"{sentence}\"\"\"

{_quality_block(difficulty)}

Kurallar:
- YALNIZCA metne dayan (uydurma yok).
- Ezbere dayalı madde numarası / tarih / rakam sorma.
- Tek doğru seçenek olmalı.
- Şıklar birbirine yakın zorlukta ve makul olmalı.
- "Hepsi/Hiçbiri" YASAK.
- Çıktı SADECE JSON olacak. Markdown / açıklama / ekstra metin YASAK.

JSON Şeması (BİREBİR):
{{
  "type": "mcq",
  "question": "...?",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct": "A",
  "explanation": "1-2 cümle kısa açıklama"
}}

SADECE JSON ÇIKTI ÜRET. JSON öncesi/sonrası hiçbir metin yazma.
""".strip()


def prompt_true_false(sentence: str, difficulty: int = 3) -> str:
    return f"""
Aşağıdaki ifadeyi temel alarak bir doğru-yanlış sorusu oluştur:

\"{sentence}\"

{_quality_block(difficulty)}

Kurallar:
- İfade önemli bir kavramı, koşulu veya çıkarımı test etmelidir.
- Tek ve net bir yargı içermelidir.
- Cümle yapısı ÇEŞİTLİ olabilir:
    - Bazı ifadeler olumlu olabilir.
    - Bazı ifadeler olumsuz (negatif) olabilir (değildir, olmaz, içermez, yapılamaz vb.).
    - Olumsuz ifade kullanımı zorunlu değildir; anlamlı olduğu yerde tercih edilmelidir.
- Anlamı bozmak için yapay olumsuzluk ekleme.
- Ezbere dayalı madde numarası soruları üretme.
- CONTEXT dışında bilgi ekleme.

Format:
Soru: ...
Cevap: Doğru / Yanlış
Açıklama: ...
""".strip()


def prompt_fill(sentence: str, difficulty: int = 3) -> str:
    rules = f"""
{_quality_block(difficulty)}

Kurallar:
- Girdi TEK bir cümledir.
- Bu cümleden yalnızca 1 boşluk doldurma sorusu üret.
- Boşluk tam olarak şu şekilde yazılmalı: _______
- Cevap:
    - cümlede aynen geçmeli
    - en fazla 3 kelime olmalı
    - soyut/genel bir kelime olmamalı
    - mümkünse teknik terim veya açık kavram olmalı
- Soru metni orijinal cümleye çok yakın kalmalı.
- Çıktı sadece JSON olacak.

JSON:
{{
  "type": "fill",
  "question": "..._______...",
  "answer": "...",
  "explanation": "..."
}}

Ek Kurallar:
- "question" içinde answer aynen görünmemeli.
- "explanation" en fazla 2 kısa cümle olmalı.
- JSON öncesi/sonrası hiçbir metin yazma.
"""

    few_shot = """
ÖRNEK 1 (İYİ):
Girdi cümle:
"Overfitting, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir."
Çıktı:
{"type":"fill", "question":"_______, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir.", "answer":"Overfitting", "explanation":"Overfitting, modelin eğitim verisine aşırı uyum sağlayıp yeni verilerde kötü performans göstermesidir."}

ÖRNEK 2 (İYİ):
Girdi cümle:
"Precision, pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriktir."
Çıktı:
{"type":"fill", "question":"Pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriğe _______ denir.", "answer":"precision", "explanation":"Precision, pozitif tahminlerin ne kadarının doğru olduğunu ölçen metriktir."}

ÖRNEK 3 (KÖTÜ/YASAK - cevap çok uzun):
{"type":"fill", "question":"...", "answer":"Modelin eğitim verisine uyum sağlaması", "explanation":"..."} <-- YASAK (cevap 3 kelimeyi aşıyor)
"""

    return f"""
{rules}

{few_shot}

Girdi cümle:
\"{sentence}\"

SADECE JSON ÇIKTI ÜRET. JSON öncesi/sonrası hiçbir metin yazma.
""".strip()


def prompt_mcq_stage1_core(context: str, difficulty: int = 3) -> str:
    return f"""
Aşağıdaki metne dayanarak 1 adet çoktan seçmeli soru için soru çekirdeği üret.

Metin:
\"{context}\"

{_quality_block(difficulty)}

Kurallar:
- JSON dışında hiçbir şey yazma.
- Soru doğrudan metindeki ana kavram, amaç, sonuç, kural, istisna, süreç veya karşılaştırma bilgisini ölçmeli.
- Soru çok genel olmamalı.
- Soru tek ve net olmalı.
- "correct_answer" kısa ve açık olmalı.
- "correct_answer" metinden çıkarılabilir olmalı.
- "rationale" 1-2 kısa cümle olmalı.
- "answer_type" şu kategorilerden biri olmalı:
  "definition" | "purpose" | "consequence" | "rule" | "exception" | "process" | "comparison"

JSON:
{{
  "question": "...?",
  "correct_answer": "...",
  "rationale": "...",
  "answer_type": "definition"
}}

Ek Kurallar:
- question mutlaka soru işareti ile bitsin.
- correct_answer 1-6 kelime arası olsun.
- Çok uzun açıklama yazma.
- JSON öncesi/sonrası hiçbir metin yazma.
""".strip()


def prompt_mcq_stage2_distractors(
    correct_answer: str,
    answer_type: str,
    rationale: str,
    context: str,
) -> str:
    return f"""
Bir MCQ sorusu için 3 adet distractor (yanlış ama makul) seçenek üret.

Bağlam:
\"{context}\"

Doğru cevap: "{correct_answer}"
Cevap tipi: "{answer_type}"
Gerekçe: "{rationale}"

Kurallar:
- JSON dışında hiçbir şey yazma.
- 3 distractor üret.
- Distractorlar doğru cevabın aynı format/kategori türünde olmalı.
- Aynı anlamı veren (synonym) veya doğruya aşırı yakın seçenek üretme.
- Metin dışı bilgi uydurma.
- "Hepsi/Hiçbiri" gibi seçenekler YASAK.
- Doğru cevabın:
    * eş anlamlısını,
    * yakın paraphrase'ını,
    * aynı anlamı veren yeniden yazımını
    distractor olarak ÜRETME.
- Distractorlar, doğru cevaba anlamsal olarak yakın görünse bile metne göre NET ŞEKİLDE yanlış olmalıdır.

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan.

Answer Type'a göre distractor stratejisi:

- answer_type == "definition":
  * kavram gibi görünmeli, ancak tanımı yanlış/eksik olmalı

- answer_type == "purpose":
  * benzer amaçlar içermeli, fakat metindeki asıl amacı yansıtmamalı

- answer_type == "consequence":
  * olası sonuç gibi görünmeli, ancak şartları yanlış yansıtmalı

- answer_type == "rule":
  * kural formatında olmalı, fakat şartları yanlış yansıtmalı

- answer_type == "exception":
  * kural KAPSAMI İÇİNDE kalan örnekler olmalı
  * doğru cevap ise kapsam DIŞI olmalı

- answer_type == "process":
  * sürecin yanlış adımı veya sırası bozuk hali olmalı

- answer_type == "comparison":
  * karşılaştırılan unsurları karıştıran/tersleyen ifadeler olmalı

JSON (anahtar ismi aynen kullan):
{{
  "distractors": ["...", "...", "..."]
}}
- JSON öncesi veya sonrası hiçbir açıklama yazma.
- Her distractor kısa, tekil ve aynı kategori/formda olsun.
""".strip()


def prompt_mcq_stage3_verify(
    question: str,
    options: dict,
    correct_letter: str,
    context: str,
) -> str:
    options_json = json.dumps(options, ensure_ascii=False)
    return f"""
Aşağıdaki MCQ'yu kalite açısından değerlendir.

Metin:
\"{context}\"

Soru: "{question}"
Seçenekler: {options_json}
Doğru seçenek harfi: "{correct_letter}"

Kurallar:
- JSON dışında hiçbir şey yazma.
- Metne göre TEK bir doğru varsa pass=true.
- Birden fazla doğru/çok belirsiz ise pass=false.
- Şıklar çok benzer/kopya ise pass=false.

JSON (anahtar isimlerini aynen kullan):
{{
  "pass": true,
  "issues": ["..."],
  "suggestion": {{
    "fix": "none" | "regenerate_distractors" | "rewrite_question",
    "notes": "..."
  }}
}}
- JSON öncesi veya sonrası hiçbir açıklama yazma.

DİL HEDEFİ:
- Üretilen soru, seçenekler ve açıklama TAMAMEN Türkçe olmalıdır.
- İngilizce soru veya İngilizce cümle kurma.
- Teknik terimler gerekiyorsa Türkçe karşılığını kullan.
""".strip()
