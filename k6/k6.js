// ===========================================================================
// AI4IA backend — 50 CONCURRENT USERS, 1 ITERATION EACH
//
// Goal: can the backend handle 50 users concurrently, where each user's single
// session ends in generating a real 30s video (which takes ~20 min)?
//
// Model: per-vu-iterations, 50 VUs x 1 iteration, NO ramp. Each VU is a distinct
// user (from loadtest_users.json) and runs journey() exactly once:
//   1. browse (all read endpoints)
//   2. full-coverage sweep (LLM POSTs + auth-safe paths + admin-403)  [FULL_COVERAGE]
//   3. create a 30s video job -> poll to completion -> fetch result/media/assets
//
// ┌──────────────────────────────────────────────────────────────────────┐
// │ ⚠ IMPORTANT — GLOBAL JOB CAP OF 20                                     │
// │ The server caps active jobs at VIDEO_JOB_MAX_ACTIVE (default 50). With 50 │
// │ users creating jobs at once, only ~20 will be accepted (202); the rest │
// │ get 503 "System is at capacity." The script treats 503 as EXPECTED and │
// │ counts it (video_rejected_503). To truly test 50 concurrent renders,   │
// │ raise VIDEO_JOB_MAX_ACTIVE above 50 on the server first.               │
// └──────────────────────────────────────────────────────────────────────┘
//
// Auth: Authorization: Bearer <jwt> per user. X-API-Key only if -e API_KEY set.
// NOTE: JWTs expire ~15 min after minting — but a 30s render takes ~20 min, so
//       the token WILL expire mid-poll. The poll loop refreshes it via
//       /auth/refresh (using the refresh_token captured in loadtest_users.json)
//       the first time a poll gets a 401, mirroring how the real frontend now
//       stays signed in through a long render. Still regenerate
//       loadtest_users.json right before running — see mint_loadtest_users.py.
//
// VU count comes from -e VUS (NOT --vus: a scenarios block makes k6 ignore
// --vus/--duration). Run length comes from -e MAX_DURATION.
//
// HTML report: the built-in web dashboard exports one at the end. Set:
//   K6_WEB_DASHBOARD=true  K6_WEB_DASHBOARD_EXPORT=report.html
// (A machine-readable summary.json is also written via handleSummary.)
//
// 10-user smoke run (Git Bash), 40-min budget:
//   K6_WEB_DASHBOARD=true K6_WEB_DASHBOARD_EXPORT=report.html \
//     k6 run -e VUS=10 -e MAX_DURATION=40m k6_loadtest.js
//
// PowerShell:
//   $env:K6_WEB_DASHBOARD="true"; $env:K6_WEB_DASHBOARD_EXPORT="report.html"
//   k6 run -e VUS=10 -e MAX_DURATION=40m k6_loadtest.js
//
// Later, the full 50-user run:
//   K6_WEB_DASHBOARD=true K6_WEB_DASHBOARD_EXPORT=report.html \
//     k6 run -e VUS=50 -e MAX_DURATION=40m k6_loadtest.js
//
// Other flags: -e FULL_COVERAGE=false (browse+video only), -e POLL_TIMEOUT=1800,
//   -e INGEST_SYNC=true, -e VERBOSE=true
// ===========================================================================

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.1.0/index.js';

// --- Config ----------------------------------------------------------------
const BASE_URL = (__ENV.BASE_URL || 'https://ai4ia.duckdns.org').replace(/\/+$/, '');
const API_PREFIX = __ENV.API_PREFIX || '/api';
const API_KEY = __ENV.API_KEY || '';
const VUS = parseInt(__ENV.VUS || '50', 10);
const ITERATIONS = parseInt(__ENV.ITERATIONS || '1', 10);      // per VU
const TARGET_DURATION = parseInt(__ENV.TARGET_DURATION || '30', 10); // 30s video
const POLL_INTERVAL = parseInt(__ENV.POLL_INTERVAL || '15', 10);     // seconds
const POLL_TIMEOUT = parseInt(__ENV.POLL_TIMEOUT || '1500', 10);     // seconds (25 min)
const MAX_DURATION = __ENV.MAX_DURATION || '35m';
const FULL_COVERAGE = (__ENV.FULL_COVERAGE || 'true') !== 'false';
const INGEST_SYNC = __ENV.INGEST_SYNC === 'true';
const VERBOSE = __ENV.VERBOSE === 'true';

// Load-test users hold real signed tokens, so the file is never committed
// (k6/loadtest_users*.json is git-ignored). Generate it right before a run
// and point USERS_FILE at it if it lives elsewhere.
const USERS_FILE = __ENV.USERS_FILE || './loadtest_users.json';
const users = new SharedArray('loadtest_users', () => JSON.parse(open(USERS_FILE)));

// --- Metrics ---------------------------------------------------------------
const errorRate = new Rate('failed_requests');
const authErrors = new Rate('auth_401_403');
const serverErrors = new Counter('server_5xx');
const videoCreated = new Counter('video_created_202');
const videoRejected = new Counter('video_rejected_503');       // hit the global cap
const videoRateLimited = new Counter('video_rate_limited_429');
const videoCompleted = new Counter('video_completed');
const videoFailed = new Counter('video_failed');
const videoTimedOut = new Counter('video_timed_out');
const tReads = new Trend('t_reads', true);
const tAnalytics = new Trend('t_analytics_llm', true);
const tJobCreate = new Trend('t_job_create', true);
const tVideoE2E = new Trend('t_video_generation_e2e', true);   // create -> completed

// Treat 2xx-4xx as "expected" so admin-403 / auth-401 error paths don't inflate
// http_req_failed. Only 5xx and network errors count as transport failures.
http.setResponseCallback(http.expectedStatuses({ min: 200, max: 499 }));

// --- Scenario: 50 VUs, 1 iteration each, no ramp --------------------------
export const options = {
  scenarios: {
    concurrent_users: {
      executor: 'per-vu-iterations',
      vus: VUS,
      iterations: ITERATIONS,
      maxDuration: MAX_DURATION,
      exec: 'journey',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
    failed_requests: ['rate<0.05'],
    auth_401_403: ['rate<0.05'],   // relaxed: tokens may expire during the long poll
  },
  // Don't abort the whole test if a few checks fail.
  setupTimeout: '120s',
  teardownTimeout: '120s',
};

// --- Helpers ---------------------------------------------------------------
function url(path) { return `${BASE_URL}${API_PREFIX}${path}`; }
function currentUser() { return users[(__VU - 1) % users.length]; }

// Access tokens are short-lived (15 min default) but a real video render
// takes 15-25+ min, so every VU's token expires mid-poll. liveAccessToken
// overrides the token baked into loadtest_users.json once refreshAccessToken()
// mints a new one — it's a plain module-level var, not the SharedArray users
// object itself, since SharedArray entries are meant to stay read-only.
let liveAccessToken = null;

function headers(user, extra) {
  const h = { Authorization: `Bearer ${liveAccessToken || user.access_token}`, Accept: 'application/json' };
  if (API_KEY) h['X-API-Key'] = API_KEY;
  return Object.assign(h, extra || {});
}

// POST /auth/refresh using the refresh_token captured at mint time (an
// httpOnly cookie in the real login flow, so it's sent back the same way
// here — see mint_loadtest_users.py). Updates liveAccessToken in place;
// every subsequent headers() call picks it up automatically.
function refreshAccessToken(user) {
  if (!user.refresh_token) return false;
  const res = http.post(url('/auth/refresh'), null, {
    headers: { Cookie: `refresh_token=${user.refresh_token}` },
    tags: { name: 'auth_refresh_poll' },
  });
  if (res.status !== 200) {
    if (VERBOSE) console.warn(`refresh failed for ${user.username} -> ${res.status}`);
    return false;
  }
  try {
    liveAccessToken = JSON.parse(res.body).access_token;
    return true;
  } catch (e) {
    return false;
  }
}

function hit(method, path, opts) {
  opts = opts || {};
  const user = opts.user || currentUser();
  const expect = opts.expect || [200];
  const name = opts.name || path;
  const p = {
    headers: headers(user, opts.body ? { 'Content-Type': 'application/json' } : null),
    tags: { name },
    timeout: opts.timeout || '60s',
  };
  const res = http.request(method, url(path), opts.body || null, p);
  const ok = check(res, { [`${name} [${expect.join('/')}]`]: (r) => expect.includes(r.status) });
  errorRate.add(!ok);
  if ((res.status === 401 || res.status === 403) && !expect.includes(res.status)) authErrors.add(1);
  else authErrors.add(0);
  if (res.status >= 500) serverErrors.add(1);
  if (opts.trend) opts.trend.add(res.timings.duration);
  if (!ok && VERBOSE) console.warn(`${name} -> ${res.status}: ${String(res.body).slice(0, 180)}`);
  return res;
}

function hitSSE(path, opts) {
  opts = opts || {};
  const user = opts.user || currentUser();
  const res = http.get(url(path), { headers: headers(user), tags: { name: opts.name || path }, timeout: '2s' });
  const alive = res.status === 200 || res.status === 0 || res.status === 404;
  check(res, { [`${opts.name || path} alive`]: () => alive });
  errorRate.add(!alive);
}

const TERMS = ['climate', 'energy', 'policy', 'ai ethics', 'water', 'governance', 'economy', 'regulation'];

// Real video-job prompts must match the archive's actual coverage, or the
// new archive-coverage check declines the job before any rendering happens
// (topic_not_covered) — a correct rejection that would still look like a
// load-test failure. TERMS above is fine for search/analytics calls (those
// don't reject on no-match), but doVideoGeneration needs topics confirmed
// covered — these three are the exact themes a live coverage-check error
// reported the archive has (see vjob_20260925_062708_29feee74's decline).
const VIDEO_TOPICS = [
  'AI ethics, governance and policy',
  'AI in health, science and environment',
  'AI and information access, media and journalism',
];

// ===========================================================================
// The single per-user journey.
// ===========================================================================
export function journey(data) {
  const u = currentUser();
  doReads(data, u);
  if (FULL_COVERAGE) {
    doAnalytics(data, u);
    doAuthSafe(u);
    doAdmin(data, u);
  }
  doVideoGeneration(data, u);   // the main event: create 30s video + poll to done
  if (INGEST_SYNC) doIngestSync(data, u);
}

// --- 1. Browse (all read endpoints) ---------------------------------------
function doReads(data, u) {
  group('system', () => {
    check(http.get(url('/health'), { tags: { name: 'health' } }), { 'health 200': (r) => r.status === 200 });
    check(http.get(`${BASE_URL}/metrics`, { tags: { name: 'metrics' } }), { 'metrics 200': (r) => r.status === 200 });
  });

  group('dashboard', () => hit('GET', '/dashboard', { user: u, name: 'dashboard', trend: tReads }));

  group('archive', () => {
    hit('GET', '/archive/videos?limit=20', { user: u, name: 'archive_videos', trend: tReads });
    hit('GET', '/archive/speakers', { user: u, name: 'archive_speakers', trend: tReads });
    hit('GET', '/archive/themes', { user: u, name: 'archive_themes', trend: tReads });
    hit('GET', '/archive/fragments?limit=20', { user: u, name: 'archive_fragments', trend: tReads });
    const term = randomItem(TERMS);
    hit('GET', `/archive/search?q=${encodeURIComponent(term)}&limit=25`, { user: u, name: 'archive_search', trend: tReads });
    hit('POST', '/archive/semantic-search', { user: u, name: 'semantic_search', expect: [200, 503],
      body: JSON.stringify({ query: term, top_k: 10 }) });
    if (data.videoId) {
      hit('GET', `/archive/videos/${data.videoId}`, { user: u, name: 'archive_video_detail' });
      hitSSE(`/archive/videos/${data.videoId}/stream`, { user: u, name: 'archive_video_stream' });
    }
    if (data.fragmentId) {
      hit('GET', `/archive/fragments/${data.fragmentId}`, { user: u, name: 'archive_fragment_detail' });
      hit('GET', `/archive/fragments/${data.fragmentId}/preview`, { user: u, name: 'archive_fragment_preview', expect: [200, 404] });
    }
  });

  group('insights-read', () => {
    hit('GET', '/insights', { user: u, name: 'insights_list', trend: tReads });
    if (data.insightId) {
      hit('GET', `/insights/${data.insightId}`, { user: u, name: 'insight_detail' });
      hit('GET', `/insights/${data.insightId}/sources`, { user: u, name: 'insight_sources' });
      hit('GET', `/insights/${data.insightId}/provenance`, { user: u, name: 'insight_provenance' });
    }
  });

  // provenance/feed and every /ingest/* read are admin-only now (see "Close
  // anonymous and cross-user data access") — a general load-test user
  // correctly gets 403 here. Accept it alongside 200 so it's not counted as
  // a failure; still exercises the route and still catches a real 5xx.
  const ADMIN_ONLY = [200, 401, 403];
  group('provenance', () => hit('GET', '/provenance/feed', { user: u, name: 'provenance_feed', trend: tReads, expect: ADMIN_ONLY }));

  group('video-jobs-list', () => hit('GET', '/video-jobs', { user: u, name: 'video_jobs_list', trend: tReads }));

  group('ingest-read', () => {
    hit('GET', '/ingest/jobs', { user: u, name: 'ingest_jobs', trend: tReads, expect: ADMIN_ONLY });
    hit('GET', '/ingest/videos', { user: u, name: 'ingest_videos', trend: tReads, expect: ADMIN_ONLY });
    if (data.videoId) hit('GET', `/ingest/videos/${data.videoId}`, { user: u, name: 'ingest_video_detail', expect: [200, 404, 401, 403] });
    const ij = data.ingestJobId;
    if (ij) {
      hit('GET', `/ingest/jobs/${ij}`, { user: u, name: 'ingest_job_detail', expect: [200, 404, 401, 403] });
      hit('GET', `/ingest/jobs/${ij}/provenance`, { user: u, name: 'ingest_job_provenance', expect: [200, 404, 401, 403] });
      hit('GET', `/ingest/jobs/${ij}/logs`, { user: u, name: 'ingest_job_logs', expect: [200, 404, 401, 403] });
      hitSSE(`/ingest/jobs/${ij}/events`, { user: u, name: 'ingest_job_events' });
      if (data.ingestStage) {
        hit('GET', `/ingest/jobs/${ij}/logs/${data.ingestStage}`, { user: u, name: 'ingest_job_log_stage', expect: [200, 404, 401, 403] });
        hitSSE(`/ingest/jobs/${ij}/logs/${data.ingestStage}/stream`, { user: u, name: 'ingest_job_log_stream' });
      }
    }
  });
}

// --- 2a. Analytics LLM POSTs (safe, read-only, slow) -----------------------
function doAnalytics(data, u) {
  const term = randomItem(TERMS);
  const soft = [200, 422, 502, 503];
  group('analytics', () => {
    hit('POST', '/insights/generate', { user: u, name: 'insights_generate', trend: tAnalytics, expect: soft,
      body: JSON.stringify({ query: `key ideas about ${term}`, limit: 10 }) });
    if (data.speaker1 && data.speaker2) {
      hit('POST', '/compare/speakers', { user: u, name: 'compare_speakers', trend: tAnalytics, expect: soft,
        body: JSON.stringify({ speakers: [data.speaker1, data.speaker2], topic: term, limit_per_speaker: 4 }) });
    }
    hit('POST', '/synthesis/theme', { user: u, name: 'synthesis_theme', trend: tAnalytics, expect: soft,
      body: JSON.stringify({ theme: data.theme || term, limit: 10 }) });
    hit('POST', '/relay', { user: u, name: 'relay', trend: tAnalytics, expect: soft,
      body: JSON.stringify({ topic: term, limit: 10 }) });
    const mm = JSON.stringify({ query: `concepts around ${term}`, mode: 'legacy' });
    hit('POST', '/v1/concept-mindmap', { user: u, name: 'concept_mindmap', trend: tAnalytics, expect: [200, 404, 500], body: mm });
    hit('POST', '/v1/v1/concept-mindmap', { user: u, name: 'concept_mindmap_alt', trend: tAnalytics, expect: [200, 404, 500], body: mm });
  });
}

// --- 2b. Auth routes via safe error paths (no emails / accounts) -----------
function doAuthSafe(u) {
  const rnd = `${__VU}_${__ITER}`;
  group('auth', () => {
    hit('GET', '/auth/me', { user: u, name: 'auth_me' });
    hit('POST', '/auth/signup', { user: u, name: 'auth_signup', expect: [422, 400],
      body: JSON.stringify({ username: `x${rnd}`, password: 'weak', first_name: 'L', last_name: 'T', email: `x${rnd}@example.com`, persona: 'general' }) });
    hit('POST', '/auth/login', { user: u, name: 'auth_login', expect: [401, 400, 403],
      body: JSON.stringify({ username: `noone_${rnd}`, password: 'Wrong#Pass123' }) });
    hit('POST', '/auth/refresh', { user: u, name: 'auth_refresh', expect: [401] });
    hit('POST', '/auth/logout', { user: u, name: 'auth_logout', expect: [200] });
    hit('POST', '/auth/forgot-password', { user: u, name: 'auth_forgot', expect: [200],
      body: JSON.stringify({ email: `noone_${rnd}@example.com` }) });
    hit('POST', '/auth/resend-otp', { user: u, name: 'auth_resend_otp', expect: [200, 400, 404],
      body: JSON.stringify({ email: `noone_${rnd}@example.com`, purpose: 'signup' }) });
    hit('POST', '/auth/verify-otp', { user: u, name: 'auth_verify_otp', expect: [400, 404, 422],
      body: JSON.stringify({ email: `noone_${rnd}@example.com`, code: '000000', purpose: 'signup' }) });
    hit('POST', '/auth/reset-password', { user: u, name: 'auth_reset_pw', expect: [400, 404, 422],
      body: JSON.stringify({ email: `noone_${rnd}@example.com`, code: '000000', new_password: 'Str0ng#Pass!23' }) });
  });
}

// --- 2c. Admin routes — general tokens -> 403 (safe, never executes) -------
function doAdmin(data, u) {
  const uid = users[0].user_id;
  const F = [403, 401];
  group('admin', () => {
    hit('GET', '/admin/exports/archive', { user: u, name: 'admin_exports', expect: F });
    hit('POST', '/admin/reindex', { user: u, name: 'admin_reindex', expect: F, body: '{}' });
    hit('GET', '/admin/storage/space', { user: u, name: 'admin_storage_space', expect: F });
    hit('GET', '/admin/storage/directories', { user: u, name: 'admin_storage_dirs', expect: F });
    hit('GET', '/admin/storage/directories/exports', { user: u, name: 'admin_storage_dir', expect: F });
    hit('DELETE', '/admin/storage/directories/exports', { user: u, name: 'admin_del_dir', expect: F });
    hit('DELETE', '/admin/storage/object?key=none', { user: u, name: 'admin_del_object', expect: [403, 401, 400, 422] });
    hit('DELETE', '/admin/storage/videos/nonexistent', { user: u, name: 'admin_del_video', expect: F });
    hit('DELETE', '/admin/storage/exports/nonexistent', { user: u, name: 'admin_del_export', expect: F });
    hit('POST', '/admin/cleanup/local-exports', { user: u, name: 'admin_cleanup', expect: F, body: '{}' });
  });
  group('admin-users', () => {
    hit('GET', '/admin/users', { user: u, name: 'admin_users_list', expect: F });
    hit('GET', `/admin/users/${uid}`, { user: u, name: 'admin_user_get', expect: F });
    hit('POST', '/admin/users', { user: u, name: 'admin_user_create', expect: [403, 401, 422],
      body: JSON.stringify({ username: 'x', password: 'x', email: 'x@example.com', persona: 'general' }) });
    hit('PATCH', `/admin/users/${uid}`, { user: u, name: 'admin_user_patch', expect: F, body: '{}' });
    hit('POST', `/admin/users/${uid}/reset-password`, { user: u, name: 'admin_user_reset_pw', expect: F, body: '{}' });
    hit('POST', `/admin/users/${uid}/unlock`, { user: u, name: 'admin_user_unlock', expect: F, body: '{}' });
    hit('POST', `/admin/users/${uid}/deactivate`, { user: u, name: 'admin_user_deactivate', expect: F, body: '{}' });
    hit('POST', `/admin/users/${uid}/reactivate`, { user: u, name: 'admin_user_reactivate', expect: F, body: '{}' });
    hit('DELETE', `/admin/users/${uid}`, { user: u, name: 'admin_user_delete', expect: F });
  });
}

// --- 3. THE MAIN EVENT: create a 30s video and poll to completion ----------
function doVideoGeneration(data, u) {
  const startedAt = Date.now();
  const body = JSON.stringify({
    prompt: `A ${TARGET_DURATION}-second documentary about ${randomItem(VIDEO_TOPICS)}.`,
    target_duration: TARGET_DURATION,
    style: 'documentary',
    validation_mode: 'iterative_llm',
    visual_mode: 'generated_assets',
    subtitle_mode: 'external',
  });

  const res = hit('POST', '/video-jobs', {
    user: u, name: 'video_job_create', trend: tJobCreate,
    expect: [202, 429, 503], // 503 = global cap of 20 reached; 429 = per-user/rate
    body,
  });

  if (res.status === 503) { videoRejected.add(1); return; }
  if (res.status === 429) { videoRateLimited.add(1); return; }
  if (res.status !== 202) { return; }

  videoCreated.add(1);
  let jobId;
  try { jobId = JSON.parse(res.body).job_id; } catch (e) { jobId = null; }
  if (!jobId) return;

  // Touch the SSE event stream once (best effort).
  hitSSE(`/video-jobs/${jobId}/events`, { user: u, name: 'video_job_events' });

  // Poll until terminal state or timeout. Tolerates 401 (token may expire on the
  // ~20 min render) and transient 5xx without failing the whole poll.
  const maxPolls = Math.ceil(POLL_TIMEOUT / POLL_INTERVAL);
  let terminal = null;
  for (let i = 0; i < maxPolls; i++) {
    sleep(POLL_INTERVAL);
    let pr = http.get(url(`/video-jobs/${jobId}`), { headers: headers(u), tags: { name: 'video_job_poll' } });
    if (pr.status === 401 && refreshAccessToken(u)) {
      pr = http.get(url(`/video-jobs/${jobId}`), { headers: headers(u), tags: { name: 'video_job_poll' } });
    }
    if (pr.status !== 200) {
      if (VERBOSE) console.warn(`poll ${jobId} -> ${pr.status}`);
      continue;
    }
    let status;
    try { status = JSON.parse(pr.body).status; } catch (e) { status = null; }
    if (status === 'completed' || status === 'failed' || status === 'cancelled' || status === 'error') {
      terminal = status;
      break;
    }
  }

  const elapsed = Date.now() - startedAt;
  if (terminal === 'completed') {
    videoCompleted.add(1);
    tVideoE2E.add(elapsed);
  } else if (terminal) {
    videoFailed.add(1);
  } else {
    videoTimedOut.add(1);
  }
  check(null, { 'video render completed': () => terminal === 'completed' });

  // Fetch all sub-resources on the (now real) job — accept not-ready codes.
  group('video-job-artifacts', () => {
    hit('GET', `/video-jobs/${jobId}`, { user: u, name: 'video_job_detail', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/storyline`, { user: u, name: 'video_job_storyline', expect: [200, 404, 409] });
    hit('GET', `/video-jobs/${jobId}/validation-report`, { user: u, name: 'video_job_validation', expect: [200, 404, 409] });
    hit('GET', `/video-jobs/${jobId}/provenance`, { user: u, name: 'video_job_provenance', expect: [200, 404, 409] });
    hit('GET', `/video-jobs/${jobId}/result`, { user: u, name: 'video_job_result', expect: [200, 404, 409] });
    hit('GET', `/video-jobs/${jobId}/manifest`, { user: u, name: 'vm_manifest', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/timeline`, { user: u, name: 'vm_timeline', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/media`, { user: u, name: 'vm_media', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/thumbnail`, { user: u, name: 'vm_thumbnail', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/subtitles/srt`, { user: u, name: 'vm_subtitles', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/video/signed-url`, { user: u, name: 'vm_video_signed_url', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/video/download`, { user: u, name: 'vm_video_download', expect: [200, 302, 404] });
    hitSSE(`/video-jobs/${jobId}/video/stream`, { user: u, name: 'vm_video_stream' });

    const assets = http.get(url(`/video-jobs/${jobId}/assets`), { headers: headers(u), tags: { name: 'video_job_assets' } });
    check(assets, { 'assets [200/404]': (r) => [200, 404].includes(r.status) });
    let aid = 'nonexistent-asset';
    try {
      const list = JSON.parse(assets.body);
      const arr = list.assets || (Array.isArray(list) ? list : []);
      if (arr && arr.length) aid = arr[0].asset_id || arr[0].id || aid;
    } catch (e) { /* ignore */ }
    hit('GET', `/video-jobs/${jobId}/assets/${aid}/metadata`, { user: u, name: 'vm_asset_metadata', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/assets/${aid}/signed-url`, { user: u, name: 'vm_asset_signed_url', expect: [200, 404] });
    hit('GET', `/video-jobs/${jobId}/assets/${aid}/download`, { user: u, name: 'vm_asset_download', expect: [200, 302, 404] });
    hitSSE(`/video-jobs/${jobId}/assets/${aid}/stream`, { user: u, name: 'vm_asset_stream' });
  });
}

// --- Optional: trigger a real ingest sync ---------------------------------
function doIngestSync(data, u) {
  group('ingest-sync', () => {
    hit('POST', '/ingest/sync', { user: u, name: 'ingest_sync', expect: [202, 409],
      body: JSON.stringify({ video_limit: 1 }) });
    if (data.videoId) hit('POST', `/ingest/videos/${data.videoId}/sync`, { user: u, name: 'ingest_video_sync', expect: [202, 409, 404] });
  });
}

// ===========================================================================
// setup — discover real IDs once for the read/analytics endpoints.
// ===========================================================================
export function setup() {
  const u = users[0];
  const h = { headers: headers(u) };
  console.log(`Target: ${BASE_URL}${API_PREFIX}`);
  console.log(`Model: ${VUS} VUs x ${ITERATIONS} iteration | 30s video per user | maxDuration ${MAX_DURATION}`);
  console.log(`FULL_COVERAGE=${FULL_COVERAGE} INGEST_SYNC=${INGEST_SYNC} POLL_TIMEOUT=${POLL_TIMEOUT}s`);
  console.log('⚠ Server VIDEO_JOB_MAX_ACTIVE caps accepted video jobs; raise it for a full concurrent run.');

  const health = http.get(url('/health'), h);
  if (health.status !== 200) throw new Error(`Health check failed (${health.status}) — check BASE_URL.`);

  const data = {};
  const j = (res) => { try { return JSON.parse(res.body); } catch (e) { return null; } };

  const videos = j(http.get(url('/archive/videos?limit=1'), h));
  if (videos && videos.length) data.videoId = videos[0].video_id;
  const frags = j(http.get(url('/archive/fragments?limit=1'), h));
  if (frags && frags.length) data.fragmentId = frags[0].fragment_id;
  const sp = j(http.get(url('/archive/speakers'), h));
  if (sp && sp.speakers && sp.speakers.length >= 2) { data.speaker1 = sp.speakers[0].speaker; data.speaker2 = sp.speakers[1].speaker; }
  const themes = j(http.get(url('/archive/themes'), h));
  if (themes && themes.themes && themes.themes.length) data.theme = themes.themes[0].name;
  const ins = j(http.get(url('/insights'), h));
  if (ins && ins.insights && ins.insights.length) data.insightId = ins.insights[0].insight_id;
  const ij = j(http.get(url('/ingest/jobs'), h));
  const ijobs = ij && (ij.jobs || (Array.isArray(ij) ? ij : null));
  if (ijobs && ijobs.length) {
    data.ingestJobId = ijobs[0].job_id;
    const logs = j(http.get(url(`/ingest/jobs/${data.ingestJobId}/logs`), h));
    if (logs && logs.stages_with_logs && logs.stages_with_logs.length) data.ingestStage = logs.stages_with_logs[0];
  }

  console.log(`Discovered IDs: ${JSON.stringify(data)}`);
  return data;
}

// ===========================================================================
// handleSummary — end-of-test output. The HTML report itself is produced by
// the built-in web dashboard (K6_WEB_DASHBOARD_EXPORT=report.html). Here we
// also emit the console summary and a machine-readable summary.json.
// ===========================================================================
export function handleSummary(data) {
  return {
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
    'summary.json': JSON.stringify(data, null, 2),
  };
}
