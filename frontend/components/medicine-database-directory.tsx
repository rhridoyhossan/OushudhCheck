"use client";

import React, { useState, useEffect, useRef } from "react";
import gsap from "gsap";
import {
  Search,
  Heart,
  Database,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Loader2,
  X,
  Pill,
  Building2,
  Stethoscope,
  AlertTriangle,
  Clock,
  Activity,
} from "lucide-react";

export interface MedicineRecord {
  brand_id: number;
  brand_name: string;
  type: string | null;
  dosage_form: string | null;
  generic: string;
  manufacturer: string | null;
  package_container: string | null;
  drug_class: string | null;
  indication: string | null;
  dosage_description: string | null;
  interaction_description: string | null;
  side_effects_description: string | null;
  pregnancy_and_lactation_description: string | null;
  beginner_summary?: string | null;
}

const DOSAGE_FORMS = [
  "All",
  "Tablet",
  "Capsule",
  "Syrup",
  "Powder for Suspension",
  "Oral Suspension",
  "Ophthalmic Solution",
  "IM/IV Injection",
  "Cream",
];

/**
 * Parses a continuous clinical text block (such as `dosage_description` from medicines.db)
 * into clean, well-spaced sections (`{ label, content }`).
 * Handles:
 *  - Nested headings like "Tablet: Adult : 1-2 tablets..." -> "Tablet • Adult"
 *  - Missing spaces after periods like ").The tablet"
 *  - Consecutive labels without periods like "0.5 ml (40 mg) 4 to 11 months : 1.0 ml"
 */
function parseStructuredClinicalText(raw: string): Array<{
  label: string | null;
  content: string;
}> {
  let cleaned = raw
    .replace(/\r\n/g, "\n")
    .replace(/\u00a0/g, " ")
    // Fix OCR typo in database like "1 yea r:" -> "1 year:"
    .replace(/\byea\s+r\s*:/gi, "year:")
    // Ensure a space after period/closing-paren when followed immediately by a capital letter
    .replace(/([.)।])\s*([A-Z\u0980-\u09FF])/g, "$1 $2")
    .trim();

  if (!cleaned) return [];

  // 1. Collapse double/nested headings like "Tablet: Adult :" or "Syrup/Suspension: Children under 3 months :"
  // into a single combined badge "Tablet • Adult:"
  cleaned = cleaned.replace(
    /([A-Z\u0980-\u09FF][A-Za-z0-9\u0980-\u09FF\s/()&-]{1,35}?)\s*:\s+([A-Z0-9\u0980-\u09FF][A-Za-z0-9\u0980-\u09FF\s/()&-]{1,42}?)\s*:\s+/g,
    "\n$1 • $2: "
  );

  // 2. Break before any heading that follows a sentence end (. । ! ?)
  cleaned = cleaned.replace(
    /([.।!?])\s+([A-Z0-9\u0980-\u09FF][^:.।!?\n]{1,58}?)\s*:\s+/g,
    "$1\n$2: "
  );

  // 3. Break before inline age/weight/form headings that follow a closing parenthesis or mg/ml unit without a period
  // e.g. "0.5 ml (40 mg) 4 to 11 months : 1.0 ml (80 mg) 7 to 2 years : 1.5 ml"
  cleaned = cleaned.replace(
    /(\b(?:mg|gm|ml|daily|hours|day)\)?|\))\s+((?:Adults?|Children|Paediatric|Infant|Neonates?|[0-9]+\s*(?:to|-)\s*[0-9]+\s*(?:months?|years?)|[0-9]+\s*(?:months?|years?))[^:\n]{0,35}?)\s*:\s+/gi,
    "$1\n$2: "
  );

  const rawLines = cleaned
    .split(/\n+/)
    .map((l) => l.trim())
    .filter(Boolean);

  const items: Array<{ label: string | null; content: string }> = [];

  for (const line of rawLines) {
    const colonMatch = line.match(
      /^([A-Z0-9\u0980-\u09FF][^:]{1,72}?)\s*:\s+(.+)$/
    );
    if (colonMatch) {
      items.push({
        label: colonMatch[1].trim(),
        content: colonMatch[2].trim(),
      });
    } else {
      items.push({
        label: null,
        content: line,
      });
    }
  }

  return items;
}

export function MedicineDatabaseDirectory() {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [selectedForm, setSelectedForm] = useState("All");
  const [page, setPage] = useState(1);
  const [items, setItems] = useState<MedicineRecord[]>([]);
  const [total, setTotal] = useState(17191);
  const [totalPages, setTotalPages] = useState(1);
  const [stats, setStats] = useState({
    totalBrands: 17191,
    totalGenerics: 1661,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [savedBrands, setSavedBrands] = useState<Record<number, boolean>>({});
  const [activeMedicine, setActiveMedicine] = useState<MedicineRecord | null>(
    null
  );
  const cardsGridRef = useRef<HTMLDivElement | null>(null);

  // One-by-one staggered card stack fade-in animation whenever database cards load
  useEffect(() => {
    if (isLoading || items.length === 0 || !cardsGridRef.current) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".db-med-card",
        {
          opacity: 0,
          y: 26,
          scale: 0.94,
          transformOrigin: "center center",
        },
        {
          opacity: 1,
          y: 0,
          scale: 1,
          duration: 0.52,
          stagger: 0.045,
          ease: "power3.out",
          clearProps: "transform,opacity",
          overwrite: "auto",
        }
      );
    }, cardsGridRef);

    return () => ctx.revert();
  }, [isLoading, items]);

  // Lock background page scroll and allow Escape key to close modal cleanly
  useEffect(() => {
    if (!activeMedicine) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActiveMedicine(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeMedicine]);

  // Debounce search input by 280ms
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchInput);
      setPage(1);
    }, 280);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    const params = new URLSearchParams({
      q: debouncedQuery,
      dosageForm: selectedForm,
      page: String(page),
      limit: "24",
    });

    Promise.resolve()
      .then(() => {
        if (active) setIsLoading(true);
        return fetch(`/api/medicines?${params.toString()}`, {
          signal: controller.signal,
        });
      })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!active || !data) return;
        setItems(data.items || []);
        setTotal(data.total ?? 0);
        setTotalPages(data.totalPages ?? 1);
        if (data.stats) {
          setStats(data.stats);
        }
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Failed to fetch medicines:", err);
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [debouncedQuery, selectedForm, page]);

  const toggleSave = (brandId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setSavedBrands((prev) => ({ ...prev, [brandId]: !prev[brandId] }));
  };

  return (
    <section className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 sm:pt-10 pb-16 space-y-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-white/85 px-4 py-1 text-xs font-bold text-[#0ea86b] shadow-2xs border border-white">
          <Database className="size-3.5" />
          {stats.totalBrands.toLocaleString("en-US")} Brands •{" "}
          {stats.totalGenerics.toLocaleString("en-US")} Generic Compounds
        </span>
        <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#09353c] mt-4">
          Bangladeshi Medicine Database
        </h1>
        <p className="text-sm text-[#46737a] mt-3.5 leading-relaxed">
          Search any commercial brand name, generic chemical ingredient,
          manufacturer, or clinical indication.
        </p>
      </div>

      {/* Frosted Glass Search & Filter Hub in Logo Teal & Emerald Theme */}
      <div className="relative rounded-[2.5rem] bg-white/65 backdrop-blur-2xl border border-white/90 p-6 sm:p-8 shadow-[0_24px_70px_rgba(11,85,99,0.11)] overflow-hidden space-y-6">
        <div className="pointer-events-none absolute top-6 -left-10 size-48 rounded-full bg-[#0b5563]/20 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-10 right-10 size-48 rounded-full bg-[#0ea86b]/20 blur-3xl" />

        {/* Centered Pill Search Bar */}
        <div className="relative z-10 flex justify-center">
          <div className="flex items-center gap-3 rounded-full bg-white/95 backdrop-blur-xl px-5 py-2.5 shadow-lg border border-white w-full max-w-xl">
            <Search className="size-4 text-[#0ea86b] shrink-0" />
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search 17,191+ BD medicines (Napa, Ace, Seclo, Paracetamol, Square)..."
              className="w-full bg-transparent text-xs sm:text-sm font-medium text-[#09353c] placeholder:text-[#46737a]/70 focus:outline-none"
            />
            {searchInput && (
              <button
                type="button"
                onClick={() => setSearchInput("")}
                className="text-xs font-bold text-[#46737a] hover:text-[#09353c] cursor-pointer"
              >
                Clear
              </button>
            )}
            <span className="flex items-center justify-center size-8 rounded-full bg-[#0ea86b] text-white shrink-0 shadow-sm">
              {isLoading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <ArrowRight className="size-4" />
              )}
            </span>
          </div>
        </div>

        {/* Dosage Form Filter Pills */}
        <div className="relative z-10 flex flex-wrap items-center justify-center gap-2">
          {DOSAGE_FORMS.map((form) => (
            <button
              key={form}
              type="button"
              onClick={() => {
                setSelectedForm(form);
                setPage(1);
              }}
              className={`rounded-full px-4 py-1.5 text-xs font-bold border transition cursor-pointer ${
                selectedForm === form
                  ? "bg-[#0b5563] text-white border-[#0b5563] shadow-xs"
                  : "bg-white/85 text-[#46737a] border-white hover:bg-white hover:text-[#09353c]"
              }`}
            >
              {form}
            </button>
          ))}
        </div>

        {/* Result Count & Page Status Bar */}
        <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 px-1 text-xs text-[#46737a]">
          <span>
            Showing{" "}
            <strong className="text-[#09353c]">
              {total.toLocaleString("en-US")}
            </strong>{" "}
            matching medicines
          </span>
          <span>
            Page <strong className="text-[#09353c]">{page}</strong> of{" "}
            <strong className="text-[#09353c]">{Math.max(1, totalPages)}</strong>
          </span>
        </div>

        {/* 4-Column White Cards Grid */}
        {isLoading ? (
          <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
            {Array.from({ length: 12 }).map((_, idx) => (
              <div
                key={idx}
                className="h-48 rounded-2xl bg-white/70 animate-pulse p-5 border border-white"
              />
            ))}
          </div>
        ) : (
          <div
            ref={cardsGridRef}
            className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2"
          >
            {items.map((med) => {
              const isSaved = !!savedBrands[med.brand_id];
              return (
                <div
                  key={med.brand_id}
                  onClick={() => setActiveMedicine(med)}
                  className="db-med-card group rounded-2xl bg-white/95 backdrop-blur-md p-5 border border-white shadow-[0_8px_25px_rgba(11,85,99,0.06)] hover:shadow-[0_14px_35px_rgba(14,168,107,0.15)] hover:-translate-y-0.5 transition-all flex flex-col justify-between cursor-pointer"
                >
                  <div className="space-y-2.5">
                    {/* Top Row: Left Teal/Emerald Circle Icon + Right Dosage Tag & Heart Icon */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] border border-teal-100 shrink-0">
                          <span className="size-3 rounded-full border-2 border-[#0ea86b] bg-[#0ea86b]/20" />
                        </div>
                        {med.dosage_form && (
                          <span className="font-mono text-[10px] font-bold text-[#0b5563] bg-[#f2f9f8] border border-teal-100 px-2.5 py-0.5 rounded-full truncate max-w-[125px]">
                            {med.dosage_form}
                          </span>
                        )}
                      </div>

                      <button
                        type="button"
                        onClick={(e) => toggleSave(med.brand_id, e)}
                        className="text-teal-300 hover:text-[#0ea86b] transition cursor-pointer"
                      >
                        <Heart
                          className={`size-4 ${
                            isSaved ? "fill-[#0ea86b] text-[#0ea86b]" : ""
                          }`}
                        />
                      </button>
                    </div>

                    {/* Brand Name & Generic Name with clear vertical spacing */}
                    <div>
                      <h3 className="font-extrabold text-base text-[#09353c] group-hover:text-[#0ea86b] transition line-clamp-1">
                        {med.brand_name}
                      </h3>
                      <p className="text-xs font-extrabold text-[#0d9488] mt-1 line-clamp-1">
                        {med.generic}
                      </p>
                    </div>

                    {/* Cleanly Separated Metadata Rows */}
                    <div className="pt-2 border-t border-teal-50 space-y-1.5 text-[11px]">
                      {med.package_container && (
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-[#46737a] shrink-0">Pack:</span>
                          <span className="font-semibold text-[#0b5563] text-right line-clamp-1">
                            {med.package_container}
                          </span>
                        </div>
                      )}

                      <div className="flex items-start justify-between gap-2">
                        <span className="text-[#46737a] shrink-0">Company:</span>
                        <span className="font-medium text-[#09353c] text-right line-clamp-1">
                          {med.manufacturer || "Pharmaceuticals BD"}
                        </span>
                      </div>

                      {med.indication && (
                        <div className="pt-1">
                          <span className="inline-block rounded-lg bg-[#f2f9f8] px-2 py-0.5 text-[10px] font-semibold text-[#0d9488] line-clamp-1">
                            {med.indication}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {!isLoading && items.length === 0 && (
          <div className="relative z-10 text-center py-12">
            <p className="text-sm font-bold text-[#09353c]">
              No matching medicine found for &ldquo;{searchInput}&rdquo;
            </p>
            <p className="text-xs text-[#46737a] mt-1">
              Try searching for Napa, Ace, Seclo, Sergel, Paracetamol, or Square.
            </p>
          </div>
        )}

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="relative z-10 pt-4 flex items-center justify-center gap-3">
            <button
              type="button"
              disabled={page <= 1 || isLoading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="inline-flex items-center gap-1.5 rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs font-bold px-5 py-2.5 border border-teal-200 shadow-2xs disabled:opacity-40 transition cursor-pointer"
            >
              <ChevronLeft className="size-4" />
              <span>Previous</span>
            </button>

            <span className="rounded-full bg-[#0b5563] text-white text-xs font-bold px-4 py-2 shadow-xs">
              {page} / {totalPages.toLocaleString("en-US")}
            </span>

            <button
              type="button"
              disabled={page >= totalPages || isLoading}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="inline-flex items-center gap-1.5 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs font-bold px-5 py-2.5 shadow-sm disabled:opacity-40 transition cursor-pointer"
            >
              <span>Next</span>
              <ChevronRight className="size-4" />
            </button>
          </div>
        )}
      </div>

      {/* =====================================================================
          CLINICAL DETAIL MODAL (Fixed Header + Smooth Inner Scroll Body)
         ===================================================================== */}
      {activeMedicine && (
        <div
          data-lenis-prevent="true"
          onClick={() => setActiveMedicine(null)}
          className="fixed inset-0 z-50 bg-[#09353c]/60 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6"
        >
          {/* Outer Modal Shell with overflow-hidden so rounded corners never glitch with scrollbar */}
          <div
            data-lenis-prevent="true"
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl max-h-[88vh] flex flex-col rounded-[2rem] bg-white shadow-2xl border border-teal-100 overflow-hidden"
          >
            {/* Sticky Non-Scrolling Modal Header */}
            <div className="shrink-0 flex items-start justify-between gap-4 border-b border-teal-100 bg-white px-6 py-5 sm:px-8">
              <div className="space-y-1.5 min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 text-[#0ea86b] px-3 py-1 text-xs font-bold border border-emerald-200">
                    <Pill className="size-3.5" />
                    {activeMedicine.dosage_form || "Medicine"}
                  </span>
                  {activeMedicine.drug_class && (
                    <span className="rounded-full bg-[#f2f9f8] text-[#0b5563] border border-teal-100 px-3 py-1 text-xs font-semibold">
                      {activeMedicine.drug_class}
                    </span>
                  )}
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-[#09353c] pt-0.5 break-words">
                  {activeMedicine.brand_name}
                </h2>
                <p className="text-sm font-bold text-[#0d9488] break-words">
                  Generic Compound: {activeMedicine.generic}
                </p>
              </div>

              <button
                type="button"
                onClick={() => setActiveMedicine(null)}
                aria-label="Close modal"
                className="flex items-center justify-center size-9 rounded-full bg-[#f2f9f8] hover:bg-[#0ea86b] text-[#09353c] hover:text-white transition cursor-pointer shrink-0"
              >
                <X className="size-4" />
              </button>
            </div>

            {/* Scrollable Modal Body: data-lenis-prevent allows mouse wheel scrolling inside modal */}
            <div
              data-lenis-prevent="true"
              onWheel={(e) => e.stopPropagation()}
              onTouchMove={(e) => e.stopPropagation()}
              className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-6 py-6 sm:px-8 space-y-5"
            >
              {/* Top 2-Column Metadata Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
                <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-1">
                  <span className="font-bold text-[#46737a] flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
                    <Building2 className="size-3.5 text-[#0ea86b]" />
                    Manufacturer
                  </span>
                  <p className="font-extrabold text-sm text-[#09353c]">
                    {activeMedicine.manufacturer || "N/A"}
                  </p>
                </div>

                <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-1">
                  <span className="font-bold text-[#46737a] flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
                    <Stethoscope className="size-3.5 text-[#0ea86b]" />
                    Pack &amp; Container
                  </span>
                  <p className="font-extrabold text-sm text-[#09353c]">
                    {activeMedicine.package_container ||
                      `${activeMedicine.dosage_form || "Standard"} Pack`}
                  </p>
                </div>
              </div>

              {/* Beginner-Friendly Summary Highlight Card */}
              {activeMedicine.beginner_summary && (
                <div className="rounded-3xl bg-gradient-to-br from-[#f2f9f8] via-emerald-50/60 to-[#f2f9f8] p-5 border-2 border-[#0ea86b]/30 shadow-xs space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                      <Pill className="size-4 text-[#0ea86b]" />
                      Easy Summary (Beginner-Friendly Guide)
                    </p>
                    <span className="rounded-full bg-white px-2.5 py-0.5 text-[10px] font-bold text-[#0b5563] border border-teal-100 shrink-0">
                      Simple Guide
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm font-medium text-[#09353c] leading-relaxed">
                    {activeMedicine.beginner_summary}
                  </p>
                </div>
              )}

              {/* Primary Indication */}
              {activeMedicine.indication && (
                <div className="rounded-2xl bg-[#f2f9f8]/60 p-4 border border-teal-100 space-y-1.5">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563] flex items-center gap-1.5">
                    <Activity className="size-3.5 text-[#0ea86b]" />
                    Primary Indication
                  </p>
                  <p className="text-xs sm:text-sm font-semibold text-[#09353c] leading-relaxed">
                    {activeMedicine.indication}
                  </p>
                </div>
              )}

              {/* Structured Dosage & Administration with Separated Sub-Cards */}
              {activeMedicine.dosage_description && (
                <div className="rounded-3xl bg-white p-5 border border-teal-200/80 shadow-2xs space-y-3.5">
                  <div className="flex items-center justify-between border-b border-teal-100 pb-2.5">
                    <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                      <Clock className="size-4 text-[#0ea86b]" />
                      Dosage &amp; Administration
                    </p>
                    <span className="text-[10px] font-semibold text-[#46737a] bg-[#f2f9f8] px-2.5 py-0.5 rounded-full">
                      Clinical Regimen
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {parseStructuredClinicalText(
                      activeMedicine.dosage_description
                    ).map((item, idx) => (
                      <div
                        key={idx}
                        className="rounded-2xl bg-[#f2f9f8]/90 p-3.5 border border-teal-100/80 space-y-1.5"
                      >
                        {item.label && (
                          <span className="inline-block rounded-lg bg-[#0b5563] text-white px-2.5 py-0.5 text-[11px] font-bold">
                            {item.label}
                          </span>
                        )}
                        <p className="text-xs sm:text-sm text-[#09353c] leading-relaxed break-words">
                          {item.content}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Drug-Drug Interactions */}
              {activeMedicine.interaction_description && (
                <div className="rounded-3xl bg-amber-50/70 p-5 border border-amber-200/90 space-y-2.5">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                    <AlertTriangle className="size-4 text-amber-600" />
                    Drug-Drug Interactions
                  </p>
                  <div className="space-y-2">
                    {parseStructuredClinicalText(
                      activeMedicine.interaction_description
                    ).map((item, idx) => (
                      <div
                        key={idx}
                        className="rounded-2xl bg-white/85 p-3 border border-amber-200/60 space-y-1"
                      >
                        {item.label && (
                          <span className="inline-block rounded-md bg-amber-100 text-amber-900 px-2 py-0.5 text-[11px] font-bold">
                            {item.label}
                          </span>
                        )}
                        <p className="text-xs sm:text-sm text-amber-950/90 leading-relaxed break-words">
                          {item.content}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Side Effects */}
              {activeMedicine.side_effects_description && (
                <div className="rounded-3xl bg-white p-5 border border-teal-100 space-y-2.5">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                    Side Effects &amp; Precautions
                  </p>
                  <p className="text-xs sm:text-sm text-[#46737a] leading-relaxed break-words">
                    {activeMedicine.side_effects_description}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
