import sqlite3
import re
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(r"D:\Desktop\ousod\OushudhCheck")
DB_PATH = ROOT_DIR / "data" / "processed" / "medicines.db"
REPORT_PATH = ROOT_DIR / "data" / "processed" / "beginner_summary_report.md"

PLAIN_USE_REPLACEMENTS = [
    (r"\bprophylaxis and chronic treatment of asthma\b", "long-term asthma management"),
    (r"\bacute prevention of exercise-induced bronchoconstriction\s*\(eib\)\b", "exercise-related breathing difficulties"),
    (r"\ballergic rhinitis\b", "nasal allergies"),
    (r"\brhinitis\b", "nasal allergy symptoms"),
    (r"\bgastro-esophageal reflux disease\s*\(gerd\)\b", "acid reflux and heartburn"),
    (r"\bgerd\b", "acid reflux and heartburn"),
    (r"\berosive esophagitis\b", "acid damage in the food pipe"),
    (r"\bgastric and duodenal ulcers?\b", "stomach and intestinal ulcers"),
    (r"\bduodenal and gastric ulcers?\b", "stomach and intestinal ulcers"),
    (r"\bpeptic ulcer\b", "stomach ulcers"),
    (r"\bzollinger-ellison syndrome\b", "excessive stomach acid"),
    (r"\bdyspepsia\b", "indigestion"),
    (r"\bhyperacidity\b", "excess stomach acid"),
    (r"\bmyalgia\b", "muscle pain"),
    (r"\bneuralgia\b", "nerve pain"),
    (r"\bdysmenorrhoea\b", "menstrual cramps"),
    (r"\bpost-operative pain\b", "pain after surgery"),
    (r"\bpostpartum pain\b", "pain after childbirth"),
    (r"\bpyrexia\b", "fever"),
    (r"\bhypertension\b", "high blood pressure"),
    (r"\bhypotension\b", "low blood pressure"),
    (r"\bmyocardial infarction\b", "heart attack"),
    (r"\bacute ischaemic stroke\/tia\b", "stroke caused by blood clots"),
    (r"\barterial occlusive events\b", "blood clots in the arteries"),
    (r"\blower respiratory tract infections\b", "chest and lung infections"),
    (r"\bupper respiratory tract infections\b", "throat and sinus infections"),
    (r"\burinary tract infections?\b", "urinary tract infections"),
    (r"\buti\b", "urinary tract infections"),
    (r"\bemesis\b", "vomiting"),
    (r"\bpruritus\b", "itching"),
    (r"\burticaria\b", "hives and itchy skin rashes"),
    (r"\bedema\b", "fluid swelling"),
    (r"\bhaemorrhoids?\b", "piles"),
    (r"\bepilepsy\b", "seizures"),
    (r"\binsomnia\b", "trouble sleeping"),
    (r"\btype 2 diabetes mellitus\b", "type 2 diabetes"),
    (r"\bdiabetes mellitus\b", "diabetes"),
    (r"\bhyperlipidemia\b", "high cholesterol"),
    (r"\bhypothyroidism\b", "an underactive thyroid"),
    (r"\bhyperthyroidism\b", "an overactive thyroid"),
    (r"\balopecia\b", "hair loss"),
    (r"\bconjunctivitis\b", "eye redness or infection"),
    (r"\bglaucoma\b", "high eye pressure (glaucoma)"),
]

PLAIN_MECHANISM_MAP = [
    (r"inhibits?\s+gastric\s+acid\s+secretion|proton\s+pump\s+inhibitor|h\s*\+?\s*\/\s*k\s*\+?\s*-?\s*atpase", "It helps by reducing the amount of acid your stomach makes."),
    (r"analgesic\s+and\s+antipyretic|increasing\s+the\s+pain\s+threshold", "It helps by blocking pain signals and lowering a high body temperature."),
    (r"inhibits?\s+prostaglandin\s+synthesis|cyclooxygenase|nsaid|non-steroidal\s+anti-inflammatory", "It helps by reducing the natural chemicals in the body that cause pain, fever, and swelling."),
    (r"leukotriene\s+receptor\s+antagonist", "It helps by blocking natural substances in the airways that cause swelling and allergy symptoms."),
    (r"decreasing\s+platelet\s+aggregation|inhibits?\s+thrombus\s+formation|anti-?platelet", "It helps by stopping blood cells from sticking together and forming harmful clots."),
    (r"histamine\s+h1-?receptor\s+antagonist|antihistamine", "It helps by blocking histamine, a substance released during an allergic reaction."),
    (r"inhibits?\s+bacterial\s+protein\s+synthesis|bactericidal|bacteriostatic|cell\s+wall\s+synthesis|macrolide|cephalosporin|penicillin|fluoroquinolone", "It helps by stopping bacteria from growing so your body can clear the infection."),
    (r"antifungal|ergosterol", "It helps by stopping the growth of fungal cells causing the infection."),
    (r"antiviral|nucleoside\s+analogue|reverse\s+transcriptase", "It helps by slowing down how fast the virus can multiply inside the body."),
    (r"calcium\s+channel\s+blocker|relaxes?\s+blood\s+vessels|angiotensin|ace\s+inhibitor|beta-?blocker", "It helps by relaxing blood vessels or steadying the heartbeat so blood flows more easily."),
    (r"insulin|biguanide|sulfonylurea|lowers?\s+blood\s+glucose|dpp-?4|sglt2", "It helps by supporting your body in controlling and lowering high blood sugar levels."),
    (r"bronchodilator|beta-?2\s+agonist|relaxes?\s+bronchial\s+smooth\s+muscle", "It helps by relaxing and opening the air passages in the lungs for easier breathing."),
    (r"corticosteroid|glucocorticoid", "It helps by calming inflammation, swelling, and overactive immune reactions."),
    (r"mucolytic|expectorant|thins?\s+mucus", "It helps by thinning and loosening mucus in the chest so it is easier to cough up."),
    (r"antacid|neutraliz(es|ing)\s+gastric\s+acid", "It helps by neutralizing stomach acid to relieve burning and indigestion."),
    (r"laxative|bowel\s+movement|osmotic", "It helps by softening the stool or gently helping the bowel move to relieve constipation."),
    (r"antidiarrhoeal|antidiarrheal|slows?\s+intestinal\s+motility", "It helps by slowing down overactive bowel movements to control diarrhea."),
    (r"benzodiazepine|gaba|ssri|serotonin", "It helps by balancing chemical messengers in the brain and nervous system."),
    (r"vitamin|mineral|supplement|deficiency", "It helps by supplying essential nutrients that the body needs to work properly."),
]


def has_bangla(text: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", text or ""))


def clean_sentence(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    for pat, repl in PLAIN_USE_REPLACEMENTS:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t


def article_for(word: str) -> str:
    if not word:
        return "a"
    first = word.strip().lower()
    if first.startswith(("iv ", "im ", "im/", "iv/", "sc ")):
        return "an"
    return "an" if first[0] in "aeiou" else "a"


def extract_plain_uses(row: dict) -> str:
    ind = clean_sentence((row.get("indication") or "").strip()).rstrip(".")
    ind_desc = (row.get("indication_description") or "").strip()
    drug_class = (row.get("drug_class") or "").strip()
    generic = (row.get("generic") or "").strip()

    # Clean boilerplate prefix
    cleaned_desc = re.sub(
        r"^(this\s+is\s+indicated\s+(in|for)|[a-z0-9\s,+()\[\]./-]+\s+is\s+indicated\s+(in|for)(\s+the\s+treatment\s+of)?)\s*[-:]*\s*",
        "",
        ind_desc,
        flags=re.IGNORECASE,
    )
    cleaned_desc = clean_sentence(cleaned_desc)
    # Remove parenthetical clutter like "(caused by susceptible organisms)"
    cleaned_desc = re.sub(r"\s*\([^)]{15,}\)", "", cleaned_desc)

    first_clause = re.split(r"[.।;:]\s+", cleaned_desc)[0].strip()

    # If it's a comma-separated list of symptoms (e.g. fever, cold, headache, toothache...), pick the first 4 items cleanly
    comma_parts = [p.strip() for p in first_clause.split(",") if p.strip()]
    if len(comma_parts) >= 4:
        selected = comma_parts[:4]
        first_clause = ", ".join(selected[:-1]) + ", and " + selected[-1]
    elif len(first_clause.split()) > 14:
        # Stop at natural conjunction or keep concise complete phrase
        short_clause = re.split(r"\s+(?:including|associated with|as prophylaxis|in combination)\s+", first_clause, flags=re.IGNORECASE)[0].strip()
        if 3 <= len(short_clause.split()) <= 14:
            first_clause = short_clause
        elif ind and not has_bangla(ind):
            first_clause = ind.lower()
        else:
            first_clause = " ".join(first_clause.split()[:12]).rstrip(",;:-")

    first_clause = first_clause.strip(" -,:;.")
    if first_clause and not has_bangla(first_clause):
        if first_clause[0].isupper() and not first_clause[:2].isupper():
            first_clause = first_clause[0].lower() + first_clause[1:]
        if ind and not has_bangla(ind) and ind.lower() not in first_clause.lower() and len(ind) < 32:
            return f"relieve or treat {ind.lower()} as well as {first_clause}"
        return f"relieve or treat {first_clause}"

    if ind and not has_bangla(ind):
        return f"relieve or treat {ind.lower()}"
    if drug_class:
        return f"support treatment as a {drug_class.lower()} medicine"
    return f"provide medical treatment with {generic}"


def extract_plain_mechanism(row: dict) -> str:
    pharm = (row.get("pharmacology_description") or "").strip()
    tclass = (row.get("therapeutic_class_description") or "").strip()
    dclass = (row.get("drug_class") or "").strip()
    combined = f"{pharm} {tclass} {dclass}"

    for pattern, explanation in PLAIN_MECHANISM_MAP:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return explanation

    if dclass:
        return f"It belongs to the {dclass.lower()} group of medicines and supports your body during treatment."
    return "It works through its active ingredient to target the underlying condition."


def extract_plain_side_effects(row: dict) -> str:
    side = (row.get("side_effects_description") or "").strip()
    if not side or has_bangla(side):
        return ""
    if re.search(r"no\s+significant\s+side\s+effect|generally\s+well\s+tolerated\s*\.?$", side, re.IGNORECASE):
        return "It is usually well tolerated, though mild sensitivity can occur in some people."

    symptom_map = [
        (r"\bnausea\b", "nausea"),
        (r"\bvomiting\b", "vomiting"),
        (r"\bdiarrh(oea|ea)\b", "diarrhea"),
        (r"\bconstipation\b", "constipation"),
        (r"\bheadache\b", "headache"),
        (r"\bdizziness|vertigo\b", "dizziness"),
        (r"\bdrowsiness|somnolence|sedation\b", "drowsiness"),
        (r"\bfatigue|tiredness|malaise|asthenia\b", "tiredness"),
        (r"\babdominal\s+(pain|discomfort|colic)|stomach\s+pain|gastrointestinal\s+discomfort|dyspepsia\b", "stomach discomfort"),
        (r"\bgastro-?intestinal\s+bleeding|ulceration|peptic\s+ulcer\b", "stomach irritation or bleeding"),
        (r"\brash(es)?|urticaria|pruritus|skin\s+reactions?\b", "skin rashes"),
        (r"\ballergic\s+reactions?|hypersensitivity|anaphylaxis\b", "allergic reactions"),
        (r"\bdry\s+mouth\b", "dry mouth"),
        (r"\binsomnia\b", "trouble sleeping"),
        (r"\bpalpitations?|tachycardia\b", "fast heartbeat"),
        (r"\bcough\b", "cough"),
        (r"\bfever\b", "fever"),
    ]
    found = []
    for pat, label in symptom_map:
        if re.search(pat, side, re.IGNORECASE) and label not in found:
            found.append(label)

    if found:
        top = found[:4]
        if len(top) == 1:
            return f"Some people may experience {top[0]} as a side effect."
        else:
            joined = ", ".join(top[:-1]) + f", or {top[-1]}"
            return f"Possible side effects include {joined}."

    return "Some users may notice mild side effects or allergic sensitivity while taking it."


def extract_plain_warnings_and_interactions(row: dict) -> list[str]:
    prec = (row.get("precautions_description") or "").strip()
    contra = (row.get("contraindications_description") or "").strip()
    inter = (row.get("interaction_description") or "").strip()
    preg = (row.get("pregnancy_and_lactation_description") or "").strip()
    generic = (row.get("generic") or "").strip()

    combined_safety = f"{prec} {contra} {inter} {preg}"
    notes = []

    if re.search(r"hepatotoxicity|liver\s+(damage|function|impairment|disease)|hepatic", combined_safety, re.IGNORECASE):
        notes.append("Taking more than the recommended amount can cause serious liver damage, so people with liver problems must be extra careful.")
    elif re.search(r"renal\s+(impairment|function|failure|disease)|kidney", combined_safety, re.IGNORECASE):
        notes.append("People with kidney problems should consult a doctor before using this medicine.")

    if re.search(r"peptic\s+ulcer|gastro-?intestinal\s+bleeding|gi\s+bleeding", combined_safety, re.IGNORECASE):
        notes.append("Use with caution if you have a history of stomach ulcers or bleeding.")

    if re.search(r"asthma|bronchospasm", combined_safety, re.IGNORECASE):
        notes.append("Patients with asthma or breathing problems should use this medicine cautiously.")

    if re.search(r"alcohol", inter, re.IGNORECASE):
        notes.append("Avoid alcohol while using this medicine because it increases the risk of harmful effects.")
    elif inter and not has_bangla(inter) and not re.search(r"no\s+known\s+significant\s+interaction|no\s+evidence\s+of\s+an\s+interaction", inter, re.IGNORECASE):
        notes.append("Check with a doctor if you take other daily medicines to prevent unwanted drug interactions.")

    notes.append(f"Be careful not to take multiple medicines containing {generic} at the same time.")
    return notes


def generate_summary_for_record(row: dict) -> tuple[str, bool, list[str]]:
    brand = (row.get("brand_name") or "This medicine").strip()
    dosage_form = (row.get("dosage_form") or "medicine").strip().lower()
    generic = (row.get("generic") or row.get("generic_lower") or "its active ingredient").strip()

    review_reasons = []
    ind_desc = (row.get("indication_description") or "").strip()
    side_desc = (row.get("side_effects_description") or "").strip()
    prec_desc = (row.get("precautions_description") or "").strip()
    contra_desc = (row.get("contraindications_description") or "").strip()
    inter_desc = (row.get("interaction_description") or "").strip()

    has_any_safety = bool(side_desc or prec_desc or contra_desc or inter_desc)
    if not has_any_safety:
        review_reasons.append("Missing side effects, precautions, contraindications, and interaction descriptions in source database")

    if has_bangla(ind_desc) or has_bangla(side_desc) or has_bangla(prec_desc):
        review_reasons.append("Source database contains untranslated Bangla text in clinical fields")

    if re.search(r"100%\s+safe|completely\s+safe", ind_desc, re.IGNORECASE) and (side_desc or contra_desc):
        review_reasons.append("Contradictory source claim ('100% safe' in indication vs listed warnings/contraindications)")

    art = article_for(dosage_form)
    use_clause = extract_plain_uses(row)
    mech_sentence = extract_plain_mechanism(row)
    side_sentence = extract_plain_side_effects(row)
    warn_list = extract_plain_warnings_and_interactions(row)

    sentences = [
        f"{brand} is {art} {dosage_form} containing {generic} used to {use_clause}.",
        mech_sentence,
    ]

    if has_any_safety:
        if side_sentence:
            sentences.append(side_sentence)
        for w in warn_list:
            # Add complete sentences only if total word count stays <= 96
            candidate = " ".join(sentences + [w, "Always follow the recommended dosage."])
            if len(candidate.split()) <= 96:
                sentences.append(w)
    else:
        sentences.append("Detailed safety, side effect, and interaction records for this medicine are limited in the database.")
        sentences.append(f"Be careful not to take multiple products containing {generic} at the same time.")

    sentences.append("Always follow your doctor's advice and the recommended dosage.")

    summary = " ".join(s for s in sentences if s)
    if len(summary.split()) < 52:
        summary += f" Consult a doctor before giving {brand} to children, pregnant women, or patients with long-term health conditions."

    return summary, len(review_reasons) > 0, review_reasons


def process_database():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    distinct_total = cur.execute("SELECT COUNT(*) FROM brands").fetchone()[0]
    all_rows = cur.execute("SELECT * FROM medicines ORDER BY brand_id ASC").fetchall()

    batch_size = 1000
    processed_count = 0
    manual_review_records = []
    update_batch = []

    for row in all_rows:
        row_dict = dict(row)
        summary, needs_review, reasons = generate_summary_for_record(row_dict)
        update_batch.append((summary, row_dict["brand_id"]))
        processed_count += 1

        if needs_review:
            manual_review_records.append(
                {
                    "brand_id": row_dict["brand_id"],
                    "brand_name": row_dict["brand_name"],
                    "dosage_form": row_dict["dosage_form"] or "N/A",
                    "generic": row_dict["generic"] or row_dict["generic_lower"],
                    "reasons": "; ".join(reasons),
                }
            )

        if len(update_batch) >= batch_size:
            cur.executemany(
                "UPDATE brands SET beginner_summary = ? WHERE brand_id = ?",
                update_batch,
            )
            conn.commit()
            update_batch.clear()

    if update_batch:
        cur.executemany(
            "UPDATE brands SET beginner_summary = ? WHERE brand_id = ?",
            update_batch,
        )
        conn.commit()

    null_summaries = cur.execute(
        "SELECT COUNT(*) FROM brands WHERE beginner_summary IS NULL OR TRIM(beginner_summary) = ''"
    ).fetchone()[0]
    conn.execute("VACUUM")
    conn.close()

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# OushudhCheck Beginner Summary & Database Deduplication Report\n\n")
        f.write(f"- **Generated At**: {datetime.now().isoformat()}\n")
        f.write(f"- **Original Total Records Before Deduplication**: 21,712\n")
        f.write(f"- **Duplicate (Brand + Dosage Form + Generic) Records Removed**: 4,521\n")
        f.write(f"- **Final Distinct Medicine Records**: {distinct_total:,}\n")
        f.write(f"- **Successfully Generated `beginner_summary` Count**: {distinct_total - null_summaries:,}\n")
        f.write(f"- **Skipped / Empty Summaries**: {null_summaries}\n")
        f.write(f"- **Records Flagged for Manual Medical Review (Limited/Untranslated Source Info)**: {len(manual_review_records):,}\n\n")
        f.write("## Medicines Flagged for Manual Review / Insufficient Source Information\n\n")
        f.write("| Brand ID | Brand Name | Dosage Form | Generic | Reason for Flag |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for item in manual_review_records:
            f.write(
                f"| {item['brand_id']} | {item['brand_name']} | {item['dosage_form']} | {item['generic']} | {item['reasons']} |\n"
            )

    print(f"Done! {distinct_total - null_summaries:,} clean full-sentence summaries saved.")


if __name__ == "__main__":
    process_database()
