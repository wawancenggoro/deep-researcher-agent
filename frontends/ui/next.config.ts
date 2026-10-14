// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from 'next'

const fileUploadMaxSizeMB = parseInt(process.env.FILE_UPLOAD_MAX_SIZE_MB || '100', 10)

// Hosts allowed to load dev-server resources (HMR) when the dev UI is opened by IP or
// domain instead of localhost, e.g. by scripts/start_e2e_public.sh. Comma-separated.
const allowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean)

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins,

  experimental: {
    serverActions: {
      bodySizeLimit: `${fileUploadMaxSizeMB}mb`,
    },
    proxyClientMaxBodySize: `${fileUploadMaxSizeMB}mb`,
  },

  turbopack: {},
}

export default nextConfig
