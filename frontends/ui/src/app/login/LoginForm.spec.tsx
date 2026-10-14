// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { describe, expect, test } from 'vitest'
import { LoginForm } from './LoginForm'

describe('LoginForm', () => {
  test('posts username, password, and next path to the gateway login endpoint', () => {
    const { container } = render(<LoginForm next="/?session=abc" />)

    const form = container.querySelector('form')
    expect(form).toHaveAttribute('method', 'post')
    expect(form).toHaveAttribute('action', '/api/login')
    expect(container.querySelector('input[name="next"]')).toHaveValue('/?session=abc')
    expect(container.querySelector('input[name="username"]')).toBeRequired()

    const password = container.querySelector('input[name="password"]')
    expect(password).toHaveAttribute('type', 'password')
    expect(password).toBeRequired()
    expect(screen.getByRole('button', { name: /sign in/i })).toHaveAttribute('type', 'submit')
  })

  test.each([
    ['invalid', 'Incorrect username or password.'],
    ['locked', 'Too many failed attempts. Try again in 15 minutes.'],
  ] as const)('shows the %s error', (error, message) => {
    render(<LoginForm error={error} next="/" />)

    expect(screen.getByRole('alert')).toHaveTextContent(message)
  })

  test('shows no error on first visit', () => {
    render(<LoginForm next="/" />)

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
