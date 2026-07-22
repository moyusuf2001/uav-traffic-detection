import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UAV Multi-Object Detection Framework",
  description: "A Robust Computer Vision Framework for Multi-Object and Vehicle Detection in UAV Imagery — Directed Study, BSU Spring 2026",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#f8fafc] min-h-screen">{children}</body>
    </html>
  );
}
