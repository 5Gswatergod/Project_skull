const API_ROOT = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === "string"
        ? payload
        : payload?.detail || "The request could not be completed.";
    throw new Error(detail);
  }

  return payload;
}

function withRepoRoot(repoRoot) {
  const query = new URLSearchParams();
  if (repoRoot?.trim()) {
    query.set("repo_root", repoRoot.trim());
  }
  const text = query.toString();
  return text ? `?${text}` : "";
}

export const api = {
  dashboard(repoRoot = "") {
    return request(`/dashboard${withRepoRoot(repoRoot)}`);
  },
  jobLog(jobId, repoRoot = "", maxChars = 24000) {
    const query = new URLSearchParams();
    if (repoRoot?.trim()) {
      query.set("repo_root", repoRoot.trim());
    }
    query.set("max_chars", String(maxChars));
    return request(`/jobs/${jobId}/log?${query.toString()}`);
  },
  launchTrain(payload) {
    return request("/launch/train", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  launchEval(payload) {
    return request("/launch/eval", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  launchSample(payload) {
    return request("/launch/sample", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  launchTest(payload) {
    return request("/launch/test", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  stopJob(jobId, repoRoot = "") {
    return request(`/jobs/${jobId}/stop`, {
      method: "POST",
      body: JSON.stringify({ repo_root: repoRoot }),
    });
  },
  deleteJob(jobId, repoRoot = "", deleteLogToo = false) {
    return request(`/jobs/${jobId}`, {
      method: "DELETE",
      body: JSON.stringify({
        repo_root: repoRoot,
        delete_log_too: deleteLogToo,
      }),
    });
  },
};
