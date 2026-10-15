// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Username/password login gate for the UI gateway (server.js).
 *
 * Enabled when both LOGIN_USERNAME and LOGIN_PASSWORD are set. The gateway checks
 * every HTTP request and WebSocket upgrade, including the /websocket chat
 * connection that is proxied straight to the backend and never reaches Next.js.
 * Requests without a valid session cookie are redirected to /login (pages) or
 * rejected with 401 (API calls and WebSockets).
 *
 * This gate is independent of REQUIRE_AUTH/NextAuth: it protects access to the UI
 * and does not send any identity to the backend.
 *
 * Environment:
 *   LOGIN_USERNAME, LOGIN_PASSWORD - The single account allowed to sign in
 *   LOGIN_SESSION_SECRET - Optional key for signing session cookies (falls back to
 *     NEXTAUTH_SECRET, then to a random per-process key, which signs everyone out
 *     on restart)
 *   SESSION_MAX_AGE_HOURS - Session lifetime (default 24)
 *   SECURE_COOKIES / NEXTAUTH_URL - Add the Secure flag (same rule as NextAuth)
 */

const crypto = require('crypto')

/** @typedef {Record<string, string | undefined>} LoginEnv Environment variables read by the gate */

/** @returns {LoginEnv} */
const defaultEnv = () => process.env

const LOGIN_COOKIE_NAME = 'dra_login'
const LOGIN_PATH = '/login'
const LOGIN_API_PATH = '/api/login'
const LOGOUT_API_PATH = '/api/logout'
const MAX_LOGIN_BODY_BYTES = 8 * 1024
const MAX_FAILED_ATTEMPTS = 10
const FAILED_ATTEMPT_WINDOW_MS = 15 * 60 * 1000

// Reachable before signing in: the login page itself, its assets, and NextAuth's
// endpoints that the root layout calls on every page (they can't grant access here).
const PUBLIC_PATHS = new Set([LOGIN_PATH, '/favicon.ico', '/api/health'])
const PUBLIC_PATH_PREFIXES = ['/_next/', '/api/auth/']

const processSecret = crypto.randomBytes(32).toString('hex')
const failedAttempts = new Map()

const sha256 = (value) => crypto.createHash('sha256').update(value).digest()

/** @returns {{ username: string, password: string } | null} */
const getLoginCredentials = (env = defaultEnv()) => {
  const username = env.LOGIN_USERNAME || ''
  const password = env.LOGIN_PASSWORD || ''
  return username && password ? { username, password } : null
}

const isLoginEnabled = (env = defaultEnv()) => getLoginCredentials(env) !== null

const getSessionSecret = (env = defaultEnv()) =>
  env.LOGIN_SESSION_SECRET || env.NEXTAUTH_SECRET || processSecret

const getSessionMaxAgeSeconds = (env = defaultEnv()) => {
  const hours = Number.parseInt(env.SESSION_MAX_AGE_HOURS || '', 10)
  return (Number.isFinite(hours) && hours > 0 ? hours : 24) * 60 * 60
}

const shouldUseSecureCookies = (env = defaultEnv()) => {
  if (env.SECURE_COOKIES !== undefined) return env.SECURE_COOKIES === 'true'
  return (env.NEXTAUTH_URL || '').startsWith('https://')
}

/** Constant-time comparison of submitted credentials against the configured account. */
const checkCredentials = (username, password, env = defaultEnv()) => {
  const expected = getLoginCredentials(env)
  if (!expected || typeof username !== 'string' || typeof password !== 'string') return false // pragma: allowlist secret
  // Compare fixed-length digests so neither the value nor its length leaks through timing.
  const userOk = crypto.timingSafeEqual(sha256(username), sha256(expected.username))
  const passOk = crypto.timingSafeEqual(sha256(password), sha256(expected.password))
  return userOk && passOk
}

// The signature covers the account, so changing LOGIN_USERNAME or LOGIN_PASSWORD
// invalidates every existing session.
const signSession = (expiresAt, env) => {
  const { username, password } = getLoginCredentials(env)
  return crypto
    .createHmac('sha256', getSessionSecret(env))
    .update(`${username}\n${sha256(password).toString('hex')}\n${expiresAt}`)
    .digest('base64url')
}

const createSessionToken = (env = defaultEnv(), now = Date.now()) => {
  const expiresAt = Math.floor(now / 1000) + getSessionMaxAgeSeconds(env)
  return `${expiresAt}.${signSession(expiresAt, env)}`
}

const verifySessionToken = (token, env = defaultEnv(), now = Date.now()) => {
  if (!isLoginEnabled(env) || typeof token !== 'string') return false
  const separator = token.indexOf('.')
  if (separator <= 0) return false
  const expiresAt = Number(token.slice(0, separator))
  if (!Number.isInteger(expiresAt) || expiresAt * 1000 <= now) return false
  const actual = Buffer.from(token.slice(separator + 1))
  const expected = Buffer.from(signSession(expiresAt, env))
  return actual.length === expected.length && crypto.timingSafeEqual(actual, expected)
}

const getCookie = (cookieHeader, name) => {
  if (!cookieHeader) return undefined
  for (const rawCookie of cookieHeader.split(';')) {
    const cookie = rawCookie.trim()
    const separator = cookie.indexOf('=')
    if (separator > 0 && cookie.slice(0, separator) === name) return cookie.slice(separator + 1)
  }
  return undefined
}

const isAuthenticated = (req, env = defaultEnv(), now = Date.now()) =>
  verifySessionToken(getCookie(req.headers.cookie, LOGIN_COOKIE_NAME), env, now)

const isPublicPath = (pathname) =>
  PUBLIC_PATHS.has(pathname) || PUBLIC_PATH_PREFIXES.some((prefix) => pathname.startsWith(prefix))

/** Return a same-site path to continue to after login, or '/' for anything else. */
const getSafeNextPath = (raw) => {
  if (typeof raw !== 'string' || !raw.startsWith('/') || raw.startsWith('//') || raw.startsWith('/\\')) {
    return '/'
  }
  if (raw === LOGIN_PATH || raw.startsWith(`${LOGIN_PATH}?`)) return '/'
  return raw
}

const getClientAddress = (req) => req.socket?.remoteAddress || 'unknown'

const isLockedOut = (address, now = Date.now()) => {
  const entry = failedAttempts.get(address)
  if (!entry) return false
  if (now - entry.firstAt > FAILED_ATTEMPT_WINDOW_MS) {
    failedAttempts.delete(address)
    return false
  }
  return entry.count >= MAX_FAILED_ATTEMPTS
}

const recordFailedAttempt = (address, now = Date.now()) => {
  const entry = failedAttempts.get(address)
  if (!entry || now - entry.firstAt > FAILED_ATTEMPT_WINDOW_MS) {
    failedAttempts.set(address, { count: 1, firstAt: now })
  } else {
    entry.count += 1
  }
  // Bound memory when many addresses fail.
  if (failedAttempts.size > 10000) {
    for (const [key, value] of failedAttempts) {
      if (now - value.firstAt > FAILED_ATTEMPT_WINDOW_MS) failedAttempts.delete(key)
    }
  }
}

const resetFailedAttempts = () => failedAttempts.clear()

const readBody = (req, limit) =>
  new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        reject(new Error('Request body too large'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })

const parseLoginBody = (body, contentType = '') => {
  if (contentType.includes('application/json')) {
    try {
      const data = JSON.parse(body)
      return { username: data.username, password: data.password, next: data.next }
    } catch {
      return {}
    }
  }
  const params = new URLSearchParams(body)
  return { username: params.get('username'), password: params.get('password'), next: params.get('next') }
}

const sessionCookie = (value, maxAgeSeconds, env) =>
  [
    `${LOGIN_COOKIE_NAME}=${value}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${maxAgeSeconds}`,
    ...(shouldUseSecureCookies(env) ? ['Secure'] : []),
  ].join('; ')

const redirect = (res, location, extraHeaders = {}) => {
  res.writeHead(303, { Location: location, 'Cache-Control': 'no-store', ...extraHeaders })
  res.end()
}

const loginPageUrl = (error, next) => {
  const params = new URLSearchParams()
  if (error) params.set('error', error)
  if (next && next !== '/') params.set('next', next)
  const query = params.toString()
  return query ? `${LOGIN_PATH}?${query}` : LOGIN_PATH
}

const handleLogin = async (req, res, env = defaultEnv()) => {
  if (req.method !== 'POST') {
    res.writeHead(405, { Allow: 'POST', 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'method_not_allowed' }))
    return
  }

  const address = getClientAddress(req)
  let fields
  try {
    fields = parseLoginBody(await readBody(req, MAX_LOGIN_BODY_BYTES), req.headers['content-type'])
  } catch {
    res.writeHead(413, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'request_too_large' }))
    return
  }
  const next = getSafeNextPath(fields.next)

  if (isLockedOut(address)) {
    redirect(res, loginPageUrl('locked', next))
    return
  }
  if (!checkCredentials(fields.username, fields.password, env)) {
    recordFailedAttempt(address)
    console.warn(`[Login] Failed sign-in attempt from ${address}`)
    redirect(res, loginPageUrl('invalid', next))
    return
  }

  failedAttempts.delete(address)
  redirect(res, next, {
    'Set-Cookie': sessionCookie(createSessionToken(env), getSessionMaxAgeSeconds(env), env),
  })
}

const handleLogout = (req, res, env = defaultEnv()) => {
  redirect(res, LOGIN_PATH, { 'Set-Cookie': sessionCookie('', 0, env) })
}

const wantsHtml = (req) =>
  (req.method === 'GET' || req.method === 'HEAD') && (req.headers.accept || '').includes('text/html')

/**
 * Apply the login gate to an HTTP request.
 *
 * @param {import('http').IncomingMessage} req
 * @param {import('http').ServerResponse} res
 * @param {string} pathname
 * @param {Record<string, unknown>} [query]
 * @param {LoginEnv} [env]
 * @returns {Promise<boolean>} true when the gate sent the response; false to let the
 *   request continue to Next.js.
 */
const gateHttpRequest = async (req, res, pathname, query = {}, env = defaultEnv()) => {
  if (!isLoginEnabled(env)) return false

  if (pathname === LOGIN_API_PATH) {
    await handleLogin(req, res, env)
    return true
  }
  if (pathname === LOGOUT_API_PATH) {
    handleLogout(req, res, env)
    return true
  }

  const authenticated = isAuthenticated(req, env)
  if (pathname === LOGIN_PATH && authenticated) {
    redirect(res, getSafeNextPath(query.next))
    return true
  }
  if (authenticated || isPublicPath(pathname)) return false

  if (wantsHtml(req)) {
    redirect(res, loginPageUrl(null, getSafeNextPath(req.url)))
  } else {
    res.writeHead(401, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })
    res.end(JSON.stringify({ error: 'login_required' }))
  }
  return true
}

/**
 * Apply the login gate to a WebSocket upgrade.
 *
 * @param {import('http').IncomingMessage} req
 * @param {import('stream').Duplex} socket
 * @param {string} pathname
 * @param {LoginEnv} [env]
 * @returns {boolean} true when the upgrade was rejected and the socket closed.
 */
const gateUpgrade = (req, socket, pathname, env = defaultEnv()) => {
  if (!isLoginEnabled(env) || isAuthenticated(req, env) || pathname.startsWith('/_next/')) return false
  try {
    socket.write('HTTP/1.1 401 Unauthorized\r\nConnection: close\r\nContent-Length: 0\r\n\r\n')
  } catch {}
  socket.destroy()
  return true
}

module.exports = {
  LOGIN_COOKIE_NAME,
  MAX_FAILED_ATTEMPTS,
  checkCredentials,
  createSessionToken,
  gateHttpRequest,
  gateUpgrade,
  getSafeNextPath,
  getSessionSecret,
  isLoginEnabled,
  isPublicPath,
  resetFailedAttempts,
  verifySessionToken,
}
