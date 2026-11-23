"use client";

import React, { useRef } from "react";
import { Mail, Globe, ExternalLink } from "lucide-react";
import { useParallaxZoom } from "@/components/scroll-reveal";

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
      className={className}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
      />
    </svg>
  );
}

interface TeamMember {
  name: string;
  role: string;
  email: string;
  githubUrl: string;
  githubUsername: string;
  bio: string;
}

const TEAM_MEMBERS: TeamMember[] = [
  {
    name: "Jubayer Ahmed",
    role: "Frontend & AI Product Engineer",
    email: "jubayer.ahm17@gmail.com",
    githubUrl: "https://github.com/Jubayerx17",
    githubUsername: "Jubayerx17",
    bio: "Building OushudhCheck's Next.js frontend and intuitive AI-powered medicine scanning interface.",
  },
  {
    name: "Ridoy Hossan Rony",
    role: "Backend & ML Systems Engineer",
    email: "rhridoyhossan@gmail.com",
    githubUrl: "https://github.com/rhridoyhossan",
    githubUsername: "rhridoyhossan",
    bio: "Building the YOLO26 blister detector, FastAPI inference engine, 21K+ BD drug resolver, and Bangla SLM pipeline.",
  },
];

export function AboutTeamSection() {
  const sectionRef = useRef<HTMLElement | null>(null);
  useParallaxZoom(sectionRef, "[data-reveal='scale']");

  return (
    <section
      ref={sectionRef}
      id="about"
      className="relative max-w-5xl mx-auto px-4 sm:px-6 py-16"
    >
      <div className="relative rounded-[2.5rem] bg-white/60 backdrop-blur-2xl border border-white/90 px-6 py-14 sm:px-12 shadow-[0_24px_70px_rgba(11,85,99,0.09)] overflow-hidden">
        {/* Background Perspective Grid Lines in Teal */}
        <div
          className="pointer-events-none absolute inset-0 opacity-35"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(13,148,136,0.14) 1px, transparent 1px), linear-gradient(to bottom, rgba(13,148,136,0.14) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        {/* Soft Ambient Glows */}
        <div className="pointer-events-none absolute -top-20 left-1/2 -translate-x-1/2 size-72 rounded-full bg-[#0d9488]/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 left-10 size-64 rounded-full bg-[#0ea86b]/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 right-10 size-64 rounded-full bg-[#0b5563]/15 blur-3xl" />

        {/* Top Pill Badge + Heading + Subtitle */}
        <div className="relative z-10 text-center max-w-xl mx-auto mb-20">
          <span className="inline-flex items-center justify-center rounded-full bg-[#0ea86b]/10 text-[#0b5563] border border-[#0ea86b]/30 px-4 py-1 text-xs font-bold tracking-wide shadow-2xs">
            Meet Our Team
          </span>

          <h2 className="mt-4 text-3xl sm:text-4xl font-extrabold tracking-tight text-[#09353c]">
            Design &amp; Delivery{" "}
            <span className="bg-gradient-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] bg-clip-text text-transparent">
              Team
            </span>
          </h2>

          <p className="mt-2.5 text-xs sm:text-sm text-[#46737a] leading-relaxed">
            Meet our OushudhCheck team: a synergy of AI engineering, creativity,
            and dedication, crafting medication safety together with passion and
            innovation.
          </p>
        </div>

        {/* 2-Card Notched Avatar Cutout Grid */}
        <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-14 md:gap-10 max-w-3xl mx-auto pt-6">
          {TEAM_MEMBERS.map((member) => {
            const githubAvatarUrl = `https://github.com/${member.githubUsername}.png`;

            return (
              <div
                key={member.githubUsername}
                data-reveal="scale"
                className="group relative flex flex-col items-center"
              >
                {/* Top Circular Avatar Floating inside the U-Shaped Card Notch */}
                <div className="absolute -top-14 z-20 flex items-center justify-center">
                  <div className="size-32 rounded-full bg-[#dff0ed] p-2.5 shadow-inner flex items-center justify-center">
                    <div className="relative size-full rounded-full overflow-hidden border-2 border-white shadow-[0_10px_30px_rgba(11,85,99,0.22)] bg-white group-hover:scale-105 transition-transform duration-300">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={githubAvatarUrl}
                        alt={member.name}
                        className="size-full object-cover"
                      />
                    </div>
                  </div>
                </div>

                {/* Main Card Body */}
                <div className="w-full rounded-[1.75rem] bg-white/95 backdrop-blur-xl pt-22 pb-7 px-6 text-center border border-teal-200/80 group-hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] group-hover:shadow-[0_20px_50px_rgba(14,168,107,0.15)] transition-all duration-300 flex flex-col items-center">
                  <h3 className="text-lg font-extrabold text-[#09353c] tracking-tight">
                    {member.name}
                  </h3>

                  <span className="mt-2 inline-block rounded-full bg-[#f2f9f8] text-[#0b5563] border border-teal-200/70 px-4 py-1 text-[11px] font-bold">
                    {member.role}
                  </span>

                  <p className="mt-3.5 text-xs text-[#46737a] leading-relaxed max-w-xs">
                    {member.bio}
                  </p>
                  <a
                    href={`mailto:${member.email}`}
                    className="mt-2 text-[11px] font-semibold text-[#0d9488] hover:text-[#0ea86b] transition"
                  >
                    {member.email}
                  </a>

                  {/* Bottom Social Icon Row */}
                  <div className="mt-5 pt-4 border-t border-teal-50 w-full flex items-center justify-center gap-4 text-[#46737a]">
                    <a
                      href={member.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`${member.name} on GitHub`}
                      className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] hover:bg-[#0b5563] hover:text-white transition"
                    >
                      <GitHubIcon className="size-4" />
                    </a>

                    <a
                      href={`mailto:${member.email}`}
                      title={`Email ${member.name}`}
                      className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] hover:bg-[#0ea86b] hover:text-white transition"
                    >
                      <Mail className="size-4" />
                    </a>

                    <a
                      href={member.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="GitHub Profile"
                      className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] hover:bg-[#0d9488] hover:text-white transition"
                    >
                      <Globe className="size-4" />
                    </a>

                    <a
                      href={member.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open GitHub"
                      className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] hover:bg-[#0b5563] hover:text-white transition"
                    >
                      <ExternalLink className="size-4" />
                    </a>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
}
