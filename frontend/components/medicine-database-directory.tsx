"use client";

import React, { useState, useEffect, useCallback } from "react";
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

export function MedicineDatabaseDirectory() {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [selectedForm, setSelectedForm] = useState("All");
  const [page, setPage] = useState(1);
  const [items, setItems] = useState<MedicineRecord[]>([]);
  const [total, setTotal] = useState(21712);
  const [totalPages, setTotalPages] = useState(1);
  const [stats, setStats] = useState({
    totalBrands: 21712,
    totalGenerics: 1711,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [savedBrands, setSavedBrands] = useState<Record<number, boolean>>({});
  const [activeMedicine, setActiveMedicine] = useState<MedicineRecord | null>(
    null
  );

  // Debounce search input by 280ms
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchInput);
      setPage(1);
    }, 280);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const fetchMedicines = useCallback(async () => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        q: debouncedQuery,
        dosageForm: selectedForm,
        page: String(page),
        limit: "24",
      });
      const res = await fetch(`/api/medicines?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setItems(data.items || []);
        setTotal(data.total ?? 0);
        setTotalPages(data.totalPages ?? 1);
        if (data.stats) {
          setStats(data.stats);
        }
      }
    } catch (err) {
      console.error("Failed to fetch medicines:", err);
    } finally {
      setIsLoading(false);
    }
  }, [debouncedQuery, selectedForm, page]);

  useEffect(() => {
    fetchMedicines();
  }, [fetchMedicines]);

  const toggleSave = (brandId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setSavedBrands((prev) => ({ ...prev, [brandId]: !prev[brandId] }));
  };

  return (
    <section className="max-w-6xl mx-auto px-4 sm:px-6 pt-4 pb-16 space-y-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-white/85 px-4 py-1 text-xs font-bold text-[#0ea86b] shadow-2xs border border-white">
          <Database className="size-3.5" />
          {stats.totalBrands.toLocaleString()} Brands •{" "}
          {stats.totalGenerics.toLocaleString()} Generic Compounds
        </span>
        <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#09353c] mt-3">
          Bangladeshi Medicine Database
        </h1>
        <p className="text-sm text-[#46737a] mt-2">
          Connected live to{" "}
          <code className="font-mono text-xs bg-white/80 px-2 py-0.5 rounded-md text-[#0b5563]">
            data/processed/medicines.db
          </code>{" "}
          — search any brand, generic chemical, manufacturer, or indication.
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
              placeholder="Search 21,712+ BD medicines (Napa, Ace, Seclo, Paracetamol, Square)..."
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
              {total.toLocaleString()}
            </strong>{" "}
            matching medicines from <code className="font-mono">medicines.db</code>
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
                className="h-44 rounded-2xl bg-white/70 animate-pulse p-5 border border-white"
              />
            ))}
          </div>
        ) : (
          <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
            {items.map((med) => {
              const isSaved = !!savedBrands[med.brand_id];
              return (
                <div
                  key={med.brand_id}
                  onClick={() => setActiveMedicine(med)}
                  className="group rounded-2xl bg-white/95 backdrop-blur-md p-5 border border-white shadow-[0_8px_25px_rgba(11,85,99,0.06)] hover:shadow-[0_14px_35px_rgba(14,168,107,0.15)] hover:-translate-y-0.5 transition-all flex flex-col justify-between cursor-pointer"
                >
                  <div>
                    {/* Top Row: Left Teal/Emerald Circle Icon + Right Dosage Tag & Heart Icon */}
                    <div className="flex items-center justify-between mb-3.5">
                      <div className="flex items-center gap-2">
                        <div className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] border border-teal-100 shrink-0">
                          <span className="size-3 rounded-full border-2 border-[#0ea86b] bg-[#0ea86b]/20" />
                        </div>
                        {med.dosage_form && (
                          <span className="font-mono text-[10px] font-bold text-[#0b5563] bg-[#f2f9f8] border border-teal-100 px-2 py-0.5 rounded-full truncate max-w-[115px]">
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

                    {/* Brand Name */}
                    <h3 className="font-extrabold text-base text-[#09353c] group-hover:text-[#0ea86b] transition line-clamp-1">
                      {med.brand_name}
                    </h3>

                    {/* Generic Name */}
                    <p className="text-xs font-extrabold text-[#0d9488] mt-1 line-clamp-1">
                      {med.generic}
                    </p>

                    {/* Package / Strength & Manufacturer */}
                    {med.package_container && (
                      <p className="text-[11px] font-semibold text-[#0b5563] mt-1 line-clamp-1">
                        {med.package_container}
                      </p>
                    )}

                    <p className="text-xs text-[#46737a] mt-1.5 line-clamp-2 leading-relaxed">
                      {med.manufacturer || "Pharmaceuticals BD"}
                      {med.indication ? ` • ${med.indication}` : ""}
                    </p>
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
              {page} / {totalPages.toLocaleString()}
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

      {/* Clinical Detail Modal when a user clicks any medicine card */}
      {activeMedicine && (
        <div
          onClick={() => setActiveMedicine(null)}
          className="fixed inset-0 z-50 bg-[#09353c]/50 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-[2rem] bg-white p-6 sm:p-8 shadow-2xl border border-teal-100 space-y-5"
          >
            <div className="flex items-start justify-between gap-4 border-b border-teal-100 pb-4">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 text-[#0ea86b] px-3 py-0.5 text-xs font-bold border border-emerald-200">
                    <Pill className="size-3.5" />
                    {activeMedicine.dosage_form || "Medicine"}
                  </span>
                  {activeMedicine.drug_class && (
                    <span className="rounded-full bg-[#f2f9f8] text-[#0b5563] px-3 py-0.5 text-xs font-semibold">
                      {activeMedicine.drug_class}
                    </span>
                  )}
                </div>
                <h2 className="text-2xl font-extrabold text-[#09353c] mt-2">
                  {activeMedicine.brand_name}
                </h2>
                <p className="text-sm font-bold text-[#0d9488]">
                  Generic: {activeMedicine.generic}
                </p>
              </div>

              <button
                type="button"
                onClick={() => setActiveMedicine(null)}
                className="flex items-center justify-center size-9 rounded-full bg-[#f2f9f8] hover:bg-teal-100 text-[#09353c] transition cursor-pointer"
              >
                <X className="size-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="rounded-2xl bg-[#f2f9f8] p-3.5 border border-teal-100/80">
                <span className="font-bold text-[#46737a] flex items-center gap-1.5">
                  <Building2 className="size-3.5 text-[#0ea86b]" />
                  Manufacturer
                </span>
                <p className="font-extrabold text-[#09353c] mt-1">
                  {activeMedicine.manufacturer || "N/A"}
                </p>
              </div>

              <div className="rounded-2xl bg-[#f2f9f8] p-3.5 border border-teal-100/80">
                <span className="font-bold text-[#46737a] flex items-center gap-1.5">
                  <Stethoscope className="size-3.5 text-[#0ea86b]" />
                  Pack &amp; Pricing Info
                </span>
                <p className="font-extrabold text-[#09353c] mt-1">
                  {activeMedicine.package_container || "Standard Pack"}
                </p>
              </div>
            </div>

            {activeMedicine.indication && (
              <div className="rounded-2xl bg-white p-4 border border-teal-100 space-y-1">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                  Primary Indication
                </p>
                <p className="text-xs text-[#46737a] leading-relaxed">
                  {activeMedicine.indication}
                </p>
              </div>
            )}

            {activeMedicine.dosage_description && (
              <div className="rounded-2xl bg-white p-4 border border-teal-100 space-y-1">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b]">
                  Dosage &amp; Administration
                </p>
                <p className="text-xs text-[#46737a] leading-relaxed whitespace-pre-line">
                  {activeMedicine.dosage_description}
                </p>
              </div>
            )}

            {activeMedicine.interaction_description && (
              <div className="rounded-2xl bg-amber-50/60 p-4 border border-amber-200 space-y-1">
                <p className="text-xs font-extrabold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                  <AlertTriangle className="size-3.5" />
                  Drug-Drug Interactions
                </p>
                <p className="text-xs text-amber-950/80 leading-relaxed">
                  {activeMedicine.interaction_description}
                </p>
              </div>
            )}

            {activeMedicine.side_effects_description && (
              <div className="rounded-2xl bg-white p-4 border border-teal-100 space-y-1">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                  Side Effects
                </p>
                <p className="text-xs text-[#46737a] leading-relaxed">
                  {activeMedicine.side_effects_description}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
