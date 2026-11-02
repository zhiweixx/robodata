# Qwen API CUDA OOM — November 2, 2026

A request to `qwen38-27b` returned **HTTP 500 with an explicit CUDA out-of-memory
error**. The request started at **07:04:25 UTC / 02:04:25 EST** on November 2;
the client recorded the failure at 07:04:53 UTC.

**[response.json](response.json) is the exact saved 797-byte response body.** It
contains no credentials, private hostname or filesystem path and is published
unchanged. Its SHA256 is
`b1c5f825d268fb8f613a1404ac6cb05a124dd95c8ae8c3a5ce2f0d2a00ba5c10`.

| Server-reported value | Amount |
|---|---:|
| GPU 0 total capacity | 139.80 GiB |
| Free GPU memory | 104.75 MiB |
| Allocation that failed | 118.00 MiB |
| Memory used by reported process 2897639 | 138.92 GiB |

The response confirms memory exhaustion on the API server. It does **not** identify
the owner of process 2897639, establish that another user caused it, or identify
the underlying serving-engine problem. The evaluation node had no cgroup OOM
event and did not trigger its RAM guard.

## Request context

| Field | Value |
|---|---|
| API | `POST /v1/chat/completions` |
| Model alias | `qwen38-27b` |
| Episode / iteration | `fill_pen_holder_s0_l0` / 11 |
| Client HTTP requests in flight at request start | 6 |
| Public-history messages / historical image blocks | 22 / 33 |
| Client output cap / automatic retries | None / 0 |
| Failed request token usage | Unknown |

Client concurrency is not the server's batch size. Historical image blocks are
counted across the submitted history, not just the latest observation. The
preceding completed request reported 60,919 prompt tokens and 2,888 completion
tokens; those numbers must not be attributed to this failed request.

[request-context.json](request-context.json) preserves the timestamps, request
hash and interpretation limits. Prompts, images, generated programs, credentials
and private connection details are excluded. Only this one error response is
published; no new API request was made for this report.

The [matching revised source snapshot](../../snapshots/qwen-api-revised220-20261102/README.md)
includes the API adapter, prompt/history logic, simulator bridge and launch code.
