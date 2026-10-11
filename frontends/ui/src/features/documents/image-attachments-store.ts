// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Image Attachments Store
 *
 * Images are uploaded to the backend but never ingested into the knowledge
 * base, so they are tracked here (shown as composer thumbnails) rather than
 * in the documents store (shown in the Files tab).
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export type ImageAttachmentStatus = 'uploading' | 'uploaded' | 'failed'

export interface ImageAttachment {
  id: string
  fileName: string
  /** Object URL for the thumbnail; revoked when the attachment is removed */
  previewUrl: string
  /** Small JPEG data URL kept with the sent message (survives refresh) */
  thumbnailUrl?: string
  status: ImageAttachmentStatus
  /** Session the image was attached in */
  collectionName: string
  errorMessage?: string
}

interface ImageAttachmentsStore {
  attachments: ImageAttachment[]
  addAttachment: (attachment: ImageAttachment) => void
  updateAttachment: (id: string, updates: Partial<ImageAttachment>) => void
  removeAttachment: (id: string) => void
  /** Remove all attachments for a session (e.g. after the message is sent) */
  clearAttachmentsForCollection: (collectionName: string) => void
}

const revokePreview = (attachment: ImageAttachment): void => {
  URL.revokeObjectURL(attachment.previewUrl)
}

export const useImageAttachmentsStore = create<ImageAttachmentsStore>()(
  devtools(
    (set) => ({
      attachments: [],

      addAttachment: (attachment) => {
        set(
          (state) => ({ attachments: [...state.attachments, attachment] }),
          false,
          'addAttachment'
        )
      },

      updateAttachment: (id, updates) => {
        set(
          (state) => ({
            attachments: state.attachments.map((a) => (a.id === id ? { ...a, ...updates } : a)),
          }),
          false,
          'updateAttachment'
        )
      },

      removeAttachment: (id) => {
        set(
          (state) => {
            const removed = state.attachments.find((a) => a.id === id)
            if (removed) revokePreview(removed)
            return { attachments: state.attachments.filter((a) => a.id !== id) }
          },
          false,
          'removeAttachment'
        )
      },

      clearAttachmentsForCollection: (collectionName) => {
        set(
          (state) => {
            const [removed, kept] = [
              state.attachments.filter((a) => a.collectionName === collectionName),
              state.attachments.filter((a) => a.collectionName !== collectionName),
            ]
            removed.forEach(revokePreview)
            return { attachments: kept }
          },
          false,
          'clearAttachmentsForCollection'
        )
      },
    }),
    { name: 'image-attachments-store' }
  )
)
