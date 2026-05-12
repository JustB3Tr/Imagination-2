const exportStatic = process.env.IMAGINATION_NEXT_EXPORT === "1"
const embedBase = (process.env.IMAGINATION_NEXT_BASE_PATH || "").replace(/\/$/, "")

/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  ...(exportStatic ? { output: "export" } : {}),
  ...(embedBase ? { basePath: embedBase, assetPrefix: embedBase } : {}),
}

export default nextConfig
