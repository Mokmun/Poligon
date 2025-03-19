import { ClerkProvider } from "@clerk/nextjs";

import localFont from "next/font/local";
import "../globals.css";


const geistSans = localFont({
  src: "../fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "../fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata = {
  title: "3D Furniture",
  description: "Generated 3d model from 2D image",
};

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html lang='en'>
        <body className="w-full h-screen flex flex-1 justify-center items-center">{children}</body>
      </html>
    </ClerkProvider>
  );
}