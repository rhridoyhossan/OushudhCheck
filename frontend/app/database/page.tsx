import React from "react";
import { Navbar } from "@/components/navbar";
import { MedicineDatabaseDirectory } from "@/components/medicine-database-directory";
import { Footer } from "@/components/footer";

export default function DatabasePage() {
  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar />
        <main>
          <MedicineDatabaseDirectory />
        </main>
      </div>
      <Footer />
    </div>
  );
}
