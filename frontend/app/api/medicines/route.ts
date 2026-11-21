import { NextRequest, NextResponse } from "next/server";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import fs from "node:fs";

const execFileAsync = promisify(execFile);

const FASTAPI_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

function resolveDatabasePath(): string {
  const candidates = [
    path.resolve(process.cwd(), "..", "data", "processed", "medicines.db"),
    path.resolve(
      process.cwd(),
      "OushudhCheck",
      "data",
      "processed",
      "medicines.db"
    ),
    "D:/Desktop/ousod/OushudhCheck/data/processed/medicines.db",
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0];
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") || "").trim();
  const dosageForm = (searchParams.get("dosageForm") || "All").trim();
  const page = Math.max(1, parseInt(searchParams.get("page") || "1", 10));
  const limit = Math.min(
    100,
    Math.max(12, parseInt(searchParams.get("limit") || "24", 10))
  );

  // 1. Primary Path: Try FastAPI Backend at http://127.0.0.1:8000/api/v1/medicines
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const fastApiRes = await fetch(
      `${FASTAPI_URL}/medicines?${searchParams.toString()}`,
      {
        signal: controller.signal,
        cache: "no-store",
      }
    );
    clearTimeout(timeout);

    if (fastApiRes.ok) {
      const data = await fastApiRes.json();
      return NextResponse.json(data);
    }
  } catch {
    // FastAPI server not started separately; fall back to direct Python read-only sqlite3 query
  }

  // 2. Fallback Path: Direct Python sqlite3 read-only query against medicines.db
  try {
    const dbPath = resolveDatabasePath();
    const pyScript = `
import sys, json, math, sqlite3
from pathlib import Path

db_path = Path(sys.argv[1])
import re as _re
q_raw = sys.argv[2].strip().lower()
q_stripped = _re.sub(r"\\b\\d+(?:\\.\\d+)?\\s*(?:mg|mcg|g|ml|iu|m)?\\b", " ", q_raw, flags=_re.IGNORECASE)
q_stripped = _re.sub(r"[*]+", " ", q_stripped)
q_clean = " ".join(q_stripped.split()).strip() or q_raw
form_clean = sys.argv[3].strip()
page = int(sys.argv[4])
limit = int(sys.argv[5])
offset = (page - 1) * limit

conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

where_clauses = []
params = []

if q_clean:
    where_clauses.append(
        "(b.brand_lower LIKE ? OR b.generic_lower LIKE ? OR LOWER(COALESCE(b.manufacturer, '')) LIKE ? OR LOWER(COALESCE(g.drug_class, '')) LIKE ? OR LOWER(COALESCE(g.indication, '')) LIKE ?)"
    )
    pat = f"%{q_clean}%"
    params.extend([pat, pat, pat, pat, pat])

if form_clean and form_clean != "All":
    where_clauses.append("b.dosage_form = ?")
    params.append(form_clean)

where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

total = cur.execute(
    f"SELECT COUNT(*) as total FROM brands b LEFT JOIN generics g ON b.generic_lower = g.generic_lower {where_sql}",
    params
).fetchone()["total"]

prefix_pat = f"{q_clean}%"
rows = cur.execute(
    f"""
    SELECT
        b.brand_id,
        b.brand_name,
        b.type,
        b.dosage_form,
        COALESCE(g.generic, b.generic_lower) as generic,
        b.manufacturer,
        b.package_container,
        g.drug_class,
        g.indication,
        g.dosage_description,
        g.interaction_description,
        g.side_effects_description,
        g.pregnancy_and_lactation_description,
        b.beginner_summary
    FROM brands b
    LEFT JOIN generics g ON b.generic_lower = g.generic_lower
    {where_sql}
    ORDER BY
        CASE
            WHEN ? != '' AND b.brand_lower = ? THEN 0
            WHEN ? != '' AND b.brand_lower LIKE ? THEN 1
            ELSE 2
        END,
        b.brand_name ASC
    LIMIT ? OFFSET ?
    """,
    [*params, q_clean, q_clean, q_clean, prefix_pat, limit, offset]
).fetchall()

stats = cur.execute(
    "SELECT (SELECT COUNT(*) FROM brands) as totalBrands, (SELECT COUNT(*) FROM generics) as totalGenerics"
).fetchone()

conn.close()

print(json.dumps({
    "items": [dict(r) for r in rows],
    "total": int(total),
    "page": page,
    "limit": limit,
    "totalPages": max(1, math.ceil(total / limit)) if total > 0 else 1,
    "stats": {
        "totalBrands": int(stats["totalBrands"]),
        "totalGenerics": int(stats["totalGenerics"])
    }
}))
`;

    const { stdout } = await execFileAsync(
      "python",
      ["-c", pyScript, dbPath, q, dosageForm, String(page), String(limit)],
      { maxBuffer: 10 * 1024 * 1024 }
    );

    const parsed = JSON.parse(stdout);
    return NextResponse.json(parsed);
  } catch (error) {
    console.error("Database query error:", error);
    return NextResponse.json(
      {
        error: "Failed to read medicines.db",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
