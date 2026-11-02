"""Keep the deployed ASPIRE API contract, replacing obsolete pilot limits."""
from deployment_docs import deployed_api_documentation


def full_api_documentation(api):
    docs = deployed_api_documentation(api)
    old = """- All actions consume the shared remaining 25Hz control ticks. A 300s Python watchdog
  guards each program; the host enforces a 1200s process deadline including startup
  and model calls. max_steps on move_to_pose limits the planned trajectory length."""
    new = """- All actions consume the task's official shared remaining 25Hz control ticks.
  There is no harness cap on program count, output tokens, or program wall time.
  Programs continue until the native task ends, the model explicitly gives up as
  a failure, or its native context is exhausted. There is no model-declared success
  action. max_steps on move_to_pose limits that trajectory length.
  Write terminating programs; observe again between motion segments as useful."""
    if docs.count(old) != 1:
        raise RuntimeError("pilot deployment contract changed; review full-budget replacement")
    return docs.replace(old, new)
