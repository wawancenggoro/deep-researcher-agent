// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Login form
 *
 * A plain HTML form that posts to /api/login (handled by the gateway in
 * login-gate.js), so signing in works without client-side JavaScript.
 */

'use client'

import { type FC } from 'react'
import { Button, Flex, FormField, Logo, Stack, Text, TextInput } from '@/adapters/ui'

export type LoginError = 'invalid' | 'locked'

const ERROR_MESSAGES: Record<LoginError, string> = {
  invalid: 'Incorrect username or password.',
  locked: 'Too many failed attempts. Try again in 15 minutes.',
}

export interface LoginFormProps {
  /** Error from the previous attempt, if any */
  error?: LoginError
  /** Path to open after signing in */
  next: string
}

export const LoginForm: FC<LoginFormProps> = ({ error, next }) => {
  return (
    <form method="post" action="/api/login">
      <input type="hidden" name="next" value={next} />
      <Stack gap="6" align="center">
        <Logo kind="horizontal" className="h-8" />

        <Flex direction="col" gap="2" align="center">
          <Text kind="title/lg">Sign in to Deep Researcher Agent</Text>
          <Text kind="body/regular/md" className="text-secondary text-center">
            Enter your username and password
          </Text>
        </Flex>

        {error && (
          <div
            role="alert"
            className="bg-status-error-muted border-status-error w-full rounded-md border p-4"
          >
            <Text kind="body/regular/sm" className="text-status-error">
              {ERROR_MESSAGES[error]}
            </Text>
          </div>
        )}

        <Stack gap="4" className="w-full">
          <FormField slotLabel="Username">
            <TextInput
              attributes={{
                TextInputValue: {
                  name: 'username',
                  autoComplete: 'username',
                  required: true,
                  autoFocus: true,
                  'aria-label': 'Username',
                },
              }}
            />
          </FormField>
          <FormField slotLabel="Password">
            <TextInput
              type="password"
              attributes={{
                TextInputValue: {
                  name: 'password',
                  autoComplete: 'current-password',
                  required: true,
                  'aria-label': 'Password',
                },
              }}
            />
          </FormField>
        </Stack>

        <Button kind="primary" size="large" type="submit" className="w-full">
          Sign in
        </Button>
      </Stack>
    </form>
  )
}
