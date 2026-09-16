import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SENTINEL-2.5D — Defense Tactical Perception System",
  description:
    "Real-time 3D LiDAR perception and threat intelligence system for defense applications · DRDO · CUDA accelerated · 25ms latency.",
  keywords: ["LiDAR", "3D", "perception", "tactical", "defense", "SENTINEL", "DRDO", "threat detection"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full bg-[#030712]">
      <body className="h-full overflow-hidden antialiased">{children}</body>
    </html>
  );
}
