import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { CustomCursor } from "@/components/custom-cursor";
import { SmoothScroll } from "@/components/smooth-scroll";
import { PageTransition } from "@/components/page-transition";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OushudhCheck (ওষুধচেক) • Multi-Strip Medicine Safety Scanner & Bangla Voice AI",
  description:
    "Real-time YOLO26 blister-pack detection, 17,000+ Bangladeshi brand-to-generic chemical resolution, Drug-Drug Interaction screening, and Bangla audio guidance.",
  icons: {
    icon: "/logo.jpg",
    shortcut: "/logo.jpg",
    apple: "/logo.jpg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} antialiased`}
    >
      <body suppressHydrationWarning className="min-h-screen flex flex-col">
        <SmoothScroll />
        <CustomCursor />
        <PageTransition>{children}</PageTransition>
        <Toaster theme="light" position="top-right" richColors />
      </body>
    </html>
  );
}
