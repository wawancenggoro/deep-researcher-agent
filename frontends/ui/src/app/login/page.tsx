// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Login Page
 *
 * Username/password sign-in for the login gate in server.js (login-gate.js),
 * enabled when LOGIN_USERNAME and LOGIN_PASSWORD are set. The form posts to
 * /api/login, which the gateway handles. Redirects home when the gate is off.
 */

import { type ReactNode } from 'react'
import { redirect } from 'next/navigation'
import { Card, Flex } from '@/adapters/ui'
import { StarfieldAnimation } from '@/shared/components/StarfieldAnimation'
import { LoginForm, type LoginError } from './LoginForm'

const isLoginEnabled = (): boolean => Boolean(process.env.LOGIN_USERNAME && process.env.LOGIN_PASSWORD)

const LOGIN_ERRORS: readonly LoginError[] = ['invalid', 'locked']

interface LoginPageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

const LoginPage = async ({ searchParams }: LoginPageProps): Promise<ReactNode> => {
  if (!isLoginEnabled()) {
    redirect('/')
  }

  const params = await searchParams
  const error = LOGIN_ERRORS.find((code) => code === params.error)
  // The gateway validates `next` again when the form is submitted.
  const next = typeof params.next === 'string' ? params.next : '/'

  return (
    <Flex
      direction="col"
      align="center"
      justify="center"
      className="bg-surface-sunken relative min-h-screen p-8"
    >
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-30">
        <div className="h-[600px] w-[600px]">
          <StarfieldAnimation particleCount={250} maxRadius={250} rotationSpeed={0.001} />
        </div>
      </div>

      <Card className="relative z-10 w-full max-w-md p-6">
        <LoginForm error={error} next={next} />
      </Card>
    </Flex>
  )
}

export default LoginPage
