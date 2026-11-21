import React from "react";
import { Navbar } from "@/components/navbar";
import { MedicineScannerWorkspace } from "@/components/medicine-scanner-workspace";
import { Footer } from "@/components/footer";

export default function ScanPage() {
  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar />
        <main>
          <MedicineScannerWorkspace />
        </main>
      </div>
      <Footer />
    </div>
  );
}
