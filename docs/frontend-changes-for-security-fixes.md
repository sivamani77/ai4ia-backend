# Frontend changes for the security fixes

The backend is closing the findings in the API & Code Validation report, and
some of those fixes change what the API accepts or returns. This page lists
what the frontend needs to change, where, and when. Backend details are in
[security-remediation.md](security-remediation.md).

Paths are relative to the frontend repo (`ai4ia/src/`). Line numbers are from
the `uat` branch at commit `c7208a8` and may have moved since.

## Summary

| # | Change | Backend PR | When | Breaks if skipped? |
|---|---|---|---|---|
| F-1 | Send the bearer token on archive search | 1 | **Before** backend PR 1 deploys | **Yes.** Archive, Relay and Speaker search return 401 |
| F-2 | Landing page reads the real speaker and theme counts | 1 | With PR 1 | No. It keeps showing the hard-coded 71 and 7 |
| F-3 | Admin Settings reads the admin dashboard | 1 | With PR 1 | Partly. Model names and storage counts go blank |
| F-4 | Concept mindmap: no change needed | 1 | — | No |
| F-5 | Refresh expired access tokens automatically | 3 | **Before** backend PR 3 deploys | **Yes.** Users are signed out every 15 minutes |
| F-6 | Handle 429 on forgot-password | 2 | With PR 2 | No. The user sees a raw error message |
| F-7 | Safari live playback puts the token in the URL | — | Existing bug | Already broken today |
| F-8 | Local development against a deployed API | 2 | With PR 2 | Only direct calls from localhost:8080 |
| F-9 | Other issues found during the review | — | Separately | No |

---

## F-1: Send the bearer token on archive search (must ship first)

**Why.** The archive now requires a signed-in user (finding A-01). One call
still goes out with plain `fetch()` and no auth header.

**Where.** `services/api.js:87`

```js
const response = await fetch(`/api/archive/search?${params.toString()}`);
```

**Change.** Use `fetchWithAuth` from `services/authApi.js:306`. It adds
`Authorization` and `X-Session-ID`.

```js
import { fetchWithAuth } from "./authApi";
...
const response = await fetchWithAuth(`${API_BASE}/archive/search?${params.toString()}`);
```

**Check.** Search on the Archive, AI Relay and Speaker pages still returns
results while signed in.

- Sending the token is harmless on the current backend, so ship this before the backend change.
- Everything else under `/api/archive/*` already sends the token: axios instances with the auth interceptor, and `useAuthedMediaUrl` for video.

---

## F-2: Landing page uses the real speaker and theme counts

**Why.** The public `GET /api/dashboard` now returns only the landing-page
counts, including two new fields:

```json
{ "archive": { "raw_videos": 310, "classified_fragments": 5112, "speakers": 307, "themes": 7 } }
```

The response no longer includes `latest`, `models`, `paths`,
`transcripts_*`, `embeddings_ready` or `embedding_matrix`.

**Where.** `pages/Home.jsx:138-160`. Focus themes and active speakers are
hard-coded:

```js
value: 7,    // focus themes
value: 71,   // active speakers
```

**Change.**

```js
value: archive.themes ?? 7,
value: archive.speakers ?? 71,
```

**Note.** The speaker figure will change from 71 to about 307. That is the
number of speakers with footage in the archive, the same list the Speakers
page shows. Product needs to confirm this is the number they want.

**Also.** `pages/ArchiveExplorer.jsx:454-456` calls `getDashboard()` and
ignores the result. Remove that call.

---

## F-3: Admin Settings reads the admin dashboard

**Why.** The full dashboard (model names, storage counts) moved to the
admin-only `GET /api/admin/dashboard`. It returns the same shape as the old
`/api/dashboard`, plus `archive.speakers` and `archive.themes`.

**Where.**
- `services/studioApi.js:185`: `export const getDashboard = () => request("/dashboard");`
- `pages/AdminSettings.jsx:310` and `:734-745`, `:1522-1528` read `models.*`, `archive.transcripts_json` and `archive.embeddings_ready`.

**Change.** Add an admin call and use it in `AdminSettings.jsx`:

```js
export const getAdminDashboard = () => request("/admin/dashboard");
```

- `pages/Dashboard.jsx:40-51` reads only `archive.raw_videos` and `archive.classified_fragments`, so it can keep using `getDashboard()`.
- `pathRows` in `AdminSettings.jsx:745` is built but never shown. It can be removed.

---

## F-4: Concept mindmap: no change needed

**What changed on the backend** (findings C-02 and C-03):
- `storyline_id` must be one of the caller's own video job ids, or the file name of one of their storylines. Anything else returns 404.
- The frontend already sends the job id (`pages/CreateVideo.jsx:633` and `:2286`), so nothing breaks.
- The response no longer contains `storyline.path`. The frontend does not read it.
- `storyline.user_prompt` is still returned, and is now always the user's own prompt. It is used in `pdf/DocumentaryReport.jsx:742-744`.
- With no `storyline_id`, the backend picks the best match among the user's own videos only. A user with no videos gets a 404, where before they got someone else's storyline. Show a friendly message for that case.

---

## F-5: Refresh expired access tokens automatically (must ship before backend PR 3)

**Why.** Access tokens currently live 30 days. That was raised from 15
minutes because the frontend had trouble with expiry. Finding C-04 requires
short-lived tokens that can be revoked. The backend already has everything
the frontend needs:
- `POST /api/auth/refresh` rotates an HttpOnly refresh cookie (path `/api/auth`, 30 days) and returns a new access token.
- An expired token gets `401 {"detail": {"code": "token_expired", "message": "Access token expired"}}`.

**What's there today.**
- `services/authApi.js:8-18` and `context/AuthContext.jsx:55-58` assume the token never expires.
- There is no global 401 handling. The only response interceptor, `studioApi.js:99-155`, handles 503 `model_warming`.
- A refresh-and-retry helper already exists as `withAuthRetry` (`authApi.js:346-365`), but only the Account Settings calls use it.
- On a 401 most pages show "Your session has expired" and stay on the page, and `VideoJobContext.jsx:372-379`, `:520-525` quietly stop polling.

**Change.**
1. **One response interceptor on every authed axios instance.** Add it inside `attachAuthInterceptors` (`authApi.js:322`). On a 401, call `/auth/refresh` once, update the token, and retry the original request once. If the refresh fails, clear the session and go to login. The instances are:
   - `studioApi` `api`
   - `adminApi`
   - `ingestApi`
   - `notificationsApi`
   - `feedbackApi`
2. **Single-flight refresh.** If several requests get a 401 at once, they should all wait on one refresh promise. Rotation revokes the old refresh token, so a second parallel refresh fails and triggers reuse detection, which ends the session.
3. **Plain `fetch` callers:**
   - `fetchWithAuth` (`authApi.js:306`)
   - SSE via `fetchEventSource` (`VideoJobContext.jsx:396-397`)
   - `fetchAuthedBlob` (`studioApi.js:176`)

   These need the same retry, and must read the token when each request is made, not when the page loads. For SSE, reconnect with the new token.
4. **hls.js:** `xhrSetup` in `LiveHlsPlayer.jsx:235-238` already reads `getAccessToken()` per request. Just confirm a refresh during playback is picked up.
5. **Retry once on `refresh_superseded`.** If two tabs refresh at the same moment, the second gets `401 {"detail": {"code": "refresh_superseded"}}`. The first tab's response has already set the new cookie, so retry the refresh once. Don't log out on this code. Only a second failure, or any other 401 from `/auth/refresh`, means the session is over.
6. **Keep the session id.** `setSession(data)` sets `sessionId = data.session_id || null`. If the refresh response has no `session_id`, keep the existing one instead of dropping `X-Session-ID`.
7. **Refresh on reload.** `bootstrapFromCache()` (`authApi.js:58-79`) should try `/auth/refresh` before giving up when the cached token is rejected.
8. Update the comments in `authApi.js:8-18`, `:318-321` and `AuthContext.jsx:55-58` that say tokens never expire.

**What the backend now does** (PR 3, on `develop`):
- Every access token carries a version. A password change, persona change or deactivation makes all of the user's existing tokens return `401 token_expired`. For a password change or deactivation the refresh also fails, so the user must sign in again. For a persona change the refresh succeeds and returns a token with the new role.
- When this backend deploys, every token issued before it returns `token_expired` once. With this interceptor in place, users won't notice.
- Persona upgrade already swaps in the returned token (`authApi.js:205`), which stays correct.

**Check.** Temporarily set `ACCESS_TOKEN_TTL_SECONDS=60` on a local backend.
Stay on the Create Video page through a render and confirm there is no
logout, SSE keeps streaming, and playback continues.

---

## F-6: Handle 429 on forgot-password

**Why.** Finding C-05: forgot-password, resend-otp, verify-otp and
reset-password are now rate-limited and can return 429 with a `Retry-After`
header in seconds. Hitting the daily code cap on forgot-password is silent:
the user still sees "If an account exists, a reset code has been sent".

**Where.** `services/authApi.js:287-290` (`forgotPassword` has no
try/catch) and `pages/Login.jsx:287`, which shows
"Request failed with status code 429".

**Change.**
- Catch 429 and show a friendly message.
- If `Retry-After` is present, show a countdown or disable the resend button for that many seconds.
- `resetPassword` (`authApi.js:299`) currently maps every 400 to the invalid-code message. Map 429 separately.

**No change needed.** The reset flow calls only forgot-password and then
reset-password (`Login.jsx:283`, `:327`). It never calls verify-otp for
resets, so the verify-otp fix (C-26) does not affect it.

---

## F-7: Safari live playback puts the token in the URL (existing bug)

**Where.** `components/LiveHlsPlayer.jsx:305-308`. Safari's native HLS sets
`video.src = masterUrl + "?access_token=" + token`.

**Problem.** The backend does not read an `access_token` query parameter, so
this already returns 401 on Safari today. Putting tokens in URLs is also
unsafe, because they end up in proxy logs and browser history.

**Change.** Use hls.js on Safari too; it is supported on macOS Safari.
Alternatively, fall back to the finished-video blob path. Don't put tokens in
URLs.

---

## F-8: Local development against a deployed API

**Why.** Finding A-14: `http://localhost:8080` is being removed from the
deployed CORS list.

**Impact.** Normal development is unaffected. The Vite dev server proxies
`/api` (`vite.config.js:24-27`, `:50-58`), so the browser only talks to
`localhost:8080`. Only code that calls `https://uat-api.ai4ia.io` directly
from the browser stops working. Keep all API calls relative to `API_BASE`.

---

## F-9: Other issues found during the review

These are not tied to a backend PR but should be tracked.

| Issue | Where | Risk | Suggested fix |
|---|---|---|---|
| Authed media blobs stay in Cache Storage after logout | `hooks/useAuthedMediaUrl.js:41-97`, cache `authed-media-cache` | On a shared machine, the next user can play the previous user's videos from cache | Clear `authed-media-cache` on logout, or key entries by user id |
| Full insight results are written to Supabase with the public key | `context/ResearchLabContext.jsx:195` → `services/studioApi.js:497-551` (`explainability_logs`) | Insight content goes to a table writable with a key shipped in the bundle | Remove, or move server-side |
| Supabase URL and publishable key hard-coded as fallbacks | `vite.config.js:43`, `services/supabaseClient.js:3-4` | The key ships in the bundle even when env config is missing | Read from env only |
| `/metrics` fetched with no auth | `services/studioApi.js:385` | Relies on `/metrics` being public (finding A-09) and will break when it is made private | Show metrics through an admin API instead |
| Password and e-mail change errors are matched on backend wording | `services/authApi.js:464-497`, `:564-600` | Breaks silently if the backend message text changes | Match on `detail.code` once the backend adds typed errors (A-05) |
| Each clip downloads the whole source video as one blob | `components/ClipPlayer.jsx:35` via `useAuthedMediaUrl` | Slow and memory-heavy for long videos | Range requests or a signed-URL endpoint for archive media |
