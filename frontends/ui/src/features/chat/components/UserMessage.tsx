// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * UserMessage Component
 *
 * User message bubble displayed in the chat area.
 */

'use client'

import { type FC } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { formatTime } from '@/shared/utils/format-time'
import type { MessageImage } from '../types'

export interface UserMessageProps {
  content: string
  /** Timestamp rendered below the bubble, right-aligned. */
  timestamp?: Date | string
  /** Images attached to the message, shown above the bubble */
  images?: MessageImage[]
}

/**
 * User message bubble component
 */
export const UserMessage: FC<UserMessageProps> = ({ content, timestamp, images }) => {
  return (
    <Flex justify="end" className="w-full">
      <Flex direction="col" align="end" className="max-w-[74%]">
        {images && images.length > 0 && (
          <Flex wrap="wrap" justify="end" gap="2" className="mb-2">
            {images.map((image) => (
              // eslint-disable-next-line @next/next/no-img-element -- data: thumbnail, next/image can't optimize it
              <img
                key={image.id}
                src={image.url}
                alt={image.fileName}
                title={image.fileName}
                className="border-base max-h-48 max-w-48 rounded-2xl border object-cover"
              />
            ))}
          </Flex>
        )}
        <div className="user-message-bubble break-words rounded-[var(--radius-card)] rounded-br-[2px] border px-4 py-3">
          <MarkdownRenderer content={content} />
        </div>
        {timestamp && (
          <Text kind="body/regular/xs" className="mono-meta text-subtle mr-1 mt-1">
            {formatTime(timestamp)}
          </Text>
        )}
      </Flex>
    </Flex>
  )
}
