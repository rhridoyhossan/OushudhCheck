import React from "react";
import { Navbar } from "@/components/navbar";
import { HeroSection } from "@/components/hero-section";
import { HealthcareShowcaseCard } from "@/components/healthcare-showcase-card";
import { ServicesGrid } from "@/components/services-grid";
import { AboutTeamSection } from "@/components/about-team-section";
import { Footer } from "@/components/footer";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar />
        <main>
          <HeroSection />
          <HealthcareShowcaseCard />
          <ServicesGrid />
          <AboutTeamSection />
        </main>
      </div>
      <Footer />
    </div>
  );
}
