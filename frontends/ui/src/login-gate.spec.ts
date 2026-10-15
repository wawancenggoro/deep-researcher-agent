// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IncomingMessage, ServerResponse } from 'node:http'
import { type Duplex, Readable } from 'node:stream'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import loginGate from '../login-gate.js'

const {
  LOGIN_COOKIE_NAME,
  MAX_FAILED_ATTEMPTS,
  checkCredentials,
  createSessionToken,
  gateHttpRequest,
  gateUpgrade,
  getSafeNextPath,
  isLoginEnabled,
  isPublicPath,
  resetFailedAttempts,
  verifySessionToken,
} = loginGate

// Test-only fixture values, not real credentials.
const TEST_PASSWORD = 's3cret' // pragma: allowlist secret
const WRONG_PASSWORD = 'wrong' // pragma: allowlist secret
const ROTATED_PASSWORD = 'rotated' // pragma: allowlist secret
const TEST_SESSION_SECRET = 'test-secret' // pragma: allowlist secret
const ENV = { LOGIN_USERNAME: 'admin', LOGIN_PASSWORD: TEST_PASSWORD, LOGIN_SESSION_SECRET: TEST_SESSION_SECRET }

interface MockRequestOptions {
  method?: string
  url?: string
  headers?: Record<string, string>
  body?: string
  address?: string
}

const createRequest = ({ method = 'GET', url = '/', headers = {}, body = '', address = '10.0.0.1' }: MockRequestOptions = {}) =>
  Object.assign(Readable.from(body ? [Buffer.from(body)] : []), {
    method,
    url,
    headers,
    socket: { remoteAddress: address },
  })

const createResponse = () => {
  const res = {
    statusCode: 0,
    headers: {} as Record<string, string>,
    body: '',
    writeHead: vi.fn((status: number, headers: Record<string, string> = {}) => {
      res.statusCode = status
      res.headers = headers
    }),
    end: vi.fn((body?: string) => {
      res.body = body ?? ''
    }),
  }
  return res
}

// The gate takes Node's request/response types; these wrappers accept the lightweight mocks.
const gate = (req: unknown, res: unknown, pathname: string, query: Record<string, unknown> = {}, env: object = ENV) =>
  gateHttpRequest(req as IncomingMessage, res as ServerResponse, pathname, query, env as Record<string, string>)

const gateWs = (req: unknown, socket: unknown, pathname: string, env: object = ENV) =>
  gateUpgrade(req as IncomingMessage, socket as Duplex, pathname, env as Record<string, string>)

const sessionCookie = (env = ENV) => `${LOGIN_COOKIE_NAME}=${createSessionToken(env)}`

const postLogin = async (fields: Record<string, string>, address = '10.0.0.1') => {
  const res = createResponse()
  const req = createRequest({
    method: 'POST',
    url: '/api/login',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(fields).toString(),
    address,
  })
  await gate(req, res, '/api/login', {}, ENV)
  return res
}

beforeEach(() => {
  resetFailedAttempts()
})

describe('isLoginEnabled', () => {
  test('requires both username and password', () => {
    expect(isLoginEnabled(ENV)).toBe(true)
    expect(isLoginEnabled({ LOGIN_USERNAME: 'admin' })).toBe(false)
    expect(isLoginEnabled({ LOGIN_PASSWORD: TEST_PASSWORD })).toBe(false)
    expect(isLoginEnabled({})).toBe(false)
  })
})

describe('checkCredentials', () => {
  test('accepts only the configured account', () => {
    expect(checkCredentials('admin', TEST_PASSWORD, ENV)).toBe(true)
    expect(checkCredentials('admin', WRONG_PASSWORD, ENV)).toBe(false)
    expect(checkCredentials('other', TEST_PASSWORD, ENV)).toBe(false)
    expect(checkCredentials(undefined, undefined, ENV)).toBe(false)
  })

  test('rejects everything when login is off', () => {
    expect(checkCredentials('admin', TEST_PASSWORD, {})).toBe(false)
  })
})

describe('session tokens', () => {
  test('a fresh token verifies', () => {
    expect(verifySessionToken(createSessionToken(ENV), ENV)).toBe(true)
  })

  test('an expired token is rejected', () => {
    const issuedAt = Date.now() - 25 * 60 * 60 * 1000
    expect(verifySessionToken(createSessionToken(ENV, issuedAt), ENV)).toBe(false)
  })

  test('a tampered token is rejected', () => {
    const [expiresAt, signature] = createSessionToken(ENV).split('.')
    expect(verifySessionToken(`${Number(expiresAt) + 3600}.${signature}`, ENV)).toBe(false)
    expect(verifySessionToken(`${expiresAt}.${signature.slice(1)}x`, ENV)).toBe(false)
    expect(verifySessionToken('not-a-token', ENV)).toBe(false)
  })

  test('changing the password signs everyone out', () => {
    const token = createSessionToken(ENV)
    expect(verifySessionToken(token, { ...ENV, LOGIN_PASSWORD: ROTATED_PASSWORD })).toBe(false)
  })
})

describe('getSafeNextPath', () => {
  test.each([
    ['/?session=abc', '/?session=abc'],
    ['/research', '/research'],
    ['https://evil.example.com', '/'],
    ['//evil.example.com', '/'],
    ['/\\evil.example.com', '/'],
    ['/login?next=/', '/'],
    [undefined, '/'],
  ])('%s -> %s', (raw, expected) => {
    expect(getSafeNextPath(raw)).toBe(expected)
  })
})

describe('isPublicPath', () => {
  test.each(['/login', '/favicon.ico', '/api/health', '/_next/static/chunk.js', '/api/auth/session'])(
    '%s is reachable before signing in',
    (path) => {
      expect(isPublicPath(path)).toBe(true)
    }
  )

  test.each(['/', '/api/chat', '/api/v1/collections', '/websocket'])('%s needs a session', (path) => {
    expect(isPublicPath(path)).toBe(false)
  })
})

describe('gateHttpRequest', () => {
  test('lets everything through when login is off', async () => {
    const res = createResponse()
    expect(await gate(createRequest(), res, '/', {}, {})).toBe(false)
    expect(res.writeHead).not.toHaveBeenCalled()
  })

  test('redirects page loads without a session to the login page', async () => {
    const res = createResponse()
    const req = createRequest({ url: '/?session=abc', headers: { accept: 'text/html' } })

    expect(await gate(req, res, '/', {}, ENV)).toBe(true)
    expect(res.statusCode).toBe(303)
    expect(res.headers.Location).toBe('/login?next=%2F%3Fsession%3Dabc')
  })

  test('rejects API calls without a session with 401', async () => {
    const res = createResponse()
    const req = createRequest({ method: 'POST', url: '/api/chat', headers: { accept: 'application/json' } })

    expect(await gate(req, res, '/api/chat', {}, ENV)).toBe(true)
    expect(res.statusCode).toBe(401)
    expect(JSON.parse(res.body)).toEqual({ error: 'login_required' })
  })

  test('lets public paths and signed-in requests through', async () => {
    const publicRes = createResponse()
    expect(await gate(createRequest({ url: '/login' }), publicRes, '/login', {}, ENV)).toBe(false)

    const signedInRes = createResponse()
    const req = createRequest({ url: '/api/chat', headers: { cookie: sessionCookie() } })
    expect(await gate(req, signedInRes, '/api/chat', {}, ENV)).toBe(false)
  })

  test('sends signed-in users away from the login page', async () => {
    const res = createResponse()
    const req = createRequest({ url: '/login?next=/research', headers: { cookie: sessionCookie() } })

    expect(await gate(req, res, '/login', { next: '/research' }, ENV)).toBe(true)
    expect(res.headers.Location).toBe('/research')
  })

  test('signs in with the right credentials', async () => {
    const res = await postLogin({ username: 'admin', password: TEST_PASSWORD, next: '/research' })

    expect(res.statusCode).toBe(303)
    expect(res.headers.Location).toBe('/research')
    const cookie = res.headers['Set-Cookie']
    expect(cookie).toMatch(new RegExp(`^${LOGIN_COOKIE_NAME}=`))
    expect(cookie).toContain('HttpOnly')
    expect(cookie).toContain('SameSite=Lax')
    const token = cookie.split(';')[0].split('=')[1]
    expect(verifySessionToken(token, ENV)).toBe(true)
  })

  test('returns to the login page with an error for wrong credentials', async () => {
    const res = await postLogin({ username: 'admin', password: WRONG_PASSWORD, next: '/research' })

    expect(res.statusCode).toBe(303)
    expect(res.headers.Location).toBe('/login?error=invalid&next=%2Fresearch')
    expect(res.headers['Set-Cookie']).toBeUndefined()
  })

  test('never redirects off-site after login', async () => {
    const res = await postLogin({ username: 'admin', password: TEST_PASSWORD, next: 'https://evil.example.com' })

    expect(res.headers.Location).toBe('/')
  })

  test('locks out an address after repeated failures', async () => {
    for (let attempt = 0; attempt < MAX_FAILED_ATTEMPTS; attempt += 1) {
      await postLogin({ username: 'admin', password: WRONG_PASSWORD }, '10.0.0.9')
    }

    const lockedOut = await postLogin({ username: 'admin', password: TEST_PASSWORD }, '10.0.0.9')
    expect(lockedOut.headers.Location).toBe('/login?error=locked')
    expect(lockedOut.headers['Set-Cookie']).toBeUndefined()

    const otherAddress = await postLogin({ username: 'admin', password: TEST_PASSWORD }, '10.0.0.10')
    expect(otherAddress.headers['Set-Cookie']).toBeDefined()
  })

  test('rejects GET on the login endpoint', async () => {
    const res = createResponse()
    await gate(createRequest({ url: '/api/login' }), res, '/api/login', {}, ENV)

    expect(res.statusCode).toBe(405)
  })

  test('signing out clears the session cookie', async () => {
    const res = createResponse()
    const req = createRequest({ method: 'POST', url: '/api/logout', headers: { cookie: sessionCookie() } })

    expect(await gate(req, res, '/api/logout', {}, ENV)).toBe(true)
    expect(res.headers.Location).toBe('/login')
    expect(res.headers['Set-Cookie']).toMatch(new RegExp(`^${LOGIN_COOKIE_NAME}=;.*Max-Age=0`))
  })
})

describe('gateUpgrade', () => {
  const createSocket = () => ({ write: vi.fn(), destroy: vi.fn() })

  test('rejects the chat WebSocket without a session', () => {
    const socket = createSocket()

    expect(gateWs(createRequest({ url: '/websocket' }), socket, '/websocket', ENV)).toBe(true)
    expect(socket.write).toHaveBeenCalledWith(expect.stringContaining('401 Unauthorized'))
    expect(socket.destroy).toHaveBeenCalled()
  })

  test('allows the chat WebSocket with a session', () => {
    const socket = createSocket()
    const req = createRequest({ url: '/websocket', headers: { cookie: sessionCookie() } })

    expect(gateWs(req, socket, '/websocket', ENV)).toBe(false)
    expect(socket.destroy).not.toHaveBeenCalled()
  })

  test('allows dev-server WebSockets and everything when login is off', () => {
    expect(gateWs(createRequest(), createSocket(), '/_next/hmr', ENV)).toBe(false)
    expect(gateWs(createRequest(), createSocket(), '/websocket', {})).toBe(false)
  })
})
