import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SENTINEL-2.5D — LiDAR Perception Prototype",
  description:
    "CUDA C++ point processing, neural segmentation and distance-adaptive LiDAR visualisation with measured performance.",
  keywords: ["LiDAR", "3D", "perception", "tactical", "defense", "SENTINEL", "Research Prototype", "threat detection"],
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
