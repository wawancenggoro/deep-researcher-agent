// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Downscale an image to a small JPEG data URL.
 *
 * Chat history lives in localStorage (a few MB total), so sent images are kept
 * as small thumbnails rather than full-size copies or blob: URLs (which don't
 * survive a page refresh).
 */

const DEFAULT_MAX_DIMENSION = 512
const JPEG_QUALITY = 0.85

const loadImage = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = src
  })

/** Returns a JPEG data URL, or undefined if the image can't be decoded */
export const createImageThumbnail = async (
  file: File,
  maxDimension = DEFAULT_MAX_DIMENSION
): Promise<string | undefined> => {
  const objectUrl = URL.createObjectURL(file)
  try {
    const img = await loadImage(objectUrl)
    const scale = Math.min(1, maxDimension / Math.max(img.naturalWidth, img.naturalHeight))
    const width = Math.max(1, Math.round(img.naturalWidth * scale))
    const height = Math.max(1, Math.round(img.naturalHeight * scale))

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    if (!ctx) return undefined

    // JPEG has no alpha; paint white so transparent PNGs don't turn black
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, width, height)
    ctx.drawImage(img, 0, 0, width, height)
    return canvas.toDataURL('image/jpeg', JPEG_QUALITY)
  } catch {
    return undefined
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}
