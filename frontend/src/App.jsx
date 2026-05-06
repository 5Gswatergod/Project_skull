import {
  startTransition,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  Activity,
  AlertTriangle,
  Blocks,
  Bot,
  Boxes,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronRight,
  Database,
  FileCode2,
  FolderKanban,
  Gauge,
  Play,
  RefreshCw,
  Rocket,
  Search,
  ShieldCheck,
  Sparkles,
  StopCircle,
  TerminalSquare,
  TestTube2,
  Trash2,
} from "lucide-react";
import { api } from "./api";

gsap.registerPlugin(useGSAP, ScrollTrigger);

const NAV_ITEMS = [
  { key: "overview", label: "Overview", icon: Gauge },
  { key: "launch", label: "Launch", icon: Rocket },
  { key: "ops", label: "Ops", icon: Activity },
  { key: "assets", label: "Assets", icon: Boxes },
  { key: "guide", label: "Guide", icon: ShieldCheck },
];

const LAUNCH_MODES = [
  { key: "train", label: "Train" },
  { key: "eval", label: "Evaluate" },
  { key: "sample", label: "Sample" },
  { key: "test", label: "Tests" },
];

const RUN_DETAIL_MODES = [
  { key: "checkpoints", label: "Checkpoints" },
  { key: "errors", label: "Errors" },
  { key: "samples", label: "Samples" },
];

const ASSET_MODES = [
  { key: "configs", label: "Configs" },
  { key: "data", label: "Data" },
  { key: "scripts", label: "Scripts" },
];

const COMMAND_GROUPS = [
  {
    title: "Launch The App",
    tone: "info",
    body: "pip install -e .[web]\npython -m skull.web",
  },
  {
    title: "Train From Config",
    tone: "accent",
    body:
      "python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml\n" +
      "python -m skull.cli.cpt --config configs/train/cpt_150m.yaml\n" +
      "python -m skull.cli.sft --config configs/train/sft_150m.yaml",
  },
  {
    title: "Eval And Sample",
    tone: "success",
    body:
      "python -m skull.cli.eval --config configs/eval/default_eval.yaml --ckpt runs/pretrain/skull_150m_base/best.pt --print_json\n" +
      "python -m skull.cli.sample --config configs/train/pretrain_150m.yaml --ckpt runs/pretrain/skull_150m_base/best.pt --prompt \"Hello\"",
  },
];

const WORKFLOW_STEPS = [
  "Clean or prepare text corpora, then verify the corpus registry is complete.",
  "Train or load a tokenizer before generating binary shards.",
  "Build bin shards and validate the run config paths before training.",
  "Launch pretraining, continued pretraining, or SFT from the dashboard or CLI.",
  "Monitor jobs, checkpoints, samples, and validation signals in Ops.",
];

function App() {
  const rootRef = useRef(null);
  const [page, setPage] = useState("overview");
  const [launchMode, setLaunchMode] = useState("train");
  const [assetMode, setAssetMode] = useState("configs");
  const [runDetailMode, setRunDetailMode] = useState("checkpoints");
  const [dashboard, setDashboard] = useState(null);
  const [repoQuery, setRepoQuery] = useState("");
  const [repoDraft, setRepoDraft] = useState("");
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedRunPath, setSelectedRunPath] = useState("");
  const [jobLog, setJobLog] = useState("");
  const [deleteLogToo, setDeleteLogToo] = useState(false);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const deferredSearch = useDeferredValue(search.trim().toLowerCase());
  const [trainForm, setTrainForm] = useState({
    config_path: "",
    requested_mode: "auto",
    use_accelerate: false,
    num_processes: "2",
    label: "",
  });
  const [evalForm, setEvalForm] = useState({
    config_path: "",
    checkpoint_path: "",
    print_json: true,
    label: "",
  });
  const [sampleForm, setSampleForm] = useState({
    config_path: "",
    checkpoint_path: "",
    prompt: "Hello from Project Skull.",
    max_new_tokens: 128,
    temperature: 1,
    top_k: "",
    label: "",
  });
  const [testForm, setTestForm] = useState({
    targets: "tests",
    extra_args: "",
    label: "tests:pytest",
  });

  const payload = dashboard ?? {
    repo_root: "",
    state: {
      summary: {},
      pipeline: [],
      configs: [],
      data_assets: { tokenizers: [], clean_files: [], bins: [], manifests: [] },
      scripts: [],
      runs: [],
      corpora: { sources: [], mixes: [] },
    },
    jobs: [],
    launchpad: { train_configs: [], eval_configs: [], checkpoints: [] },
  };

  const { state, jobs, launchpad } = payload;
  const summary = state.summary ?? {};
  const pipeline = state.pipeline ?? [];
  const activeJobs = jobs.filter((job) =>
    ["starting", "running", "stop_requested", "stopping"].includes(job.status),
  );
  const selectedJob =
    jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null;
  const selectedRun =
    state.runs?.find((run) => run.relative_path === selectedRunPath) ??
    state.runs?.[0] ??
    null;

  const refreshDashboard = useEffectEvent(async ({ quiet = false } = {}) => {
    if (!quiet) {
      setIsLoading(true);
    }
    setError("");
    try {
      const next = await api.dashboard(repoQuery);
      startTransition(() => {
        setDashboard(next);
        setRepoDraft((current) => current || next.repo_root || "");
        setSelectedJobId((current) =>
          current && next.jobs.some((job) => job.id === current)
            ? current
            : next.jobs[0]?.id || "",
        );
        setSelectedRunPath((current) =>
          current && next.state.runs.some((run) => run.relative_path === current)
            ? current
            : next.state.runs[0]?.relative_path || "",
        );
        setTrainForm((current) => ({
          ...current,
          config_path: current.config_path || next.launchpad.train_configs[0]?.path || "",
        }));
        setEvalForm((current) => ({
          ...current,
          config_path: current.config_path || next.launchpad.eval_configs[0]?.path || "",
          checkpoint_path:
            current.checkpoint_path || next.launchpad.checkpoints[0] || "",
        }));
        setSampleForm((current) => ({
          ...current,
          config_path: current.config_path || next.launchpad.train_configs[0]?.path || "",
          checkpoint_path:
            current.checkpoint_path || next.launchpad.checkpoints[0] || "",
        }));
      });
    } catch (fetchError) {
      setError(fetchError.message);
    } finally {
      setIsLoading(false);
    }
  });

  const refreshLog = useEffectEvent(async () => {
    if (!selectedJobId) {
      setJobLog("");
      return;
    }

    try {
      const next = await api.jobLog(selectedJobId, repoQuery);
      setJobLog(next.log || "");
    } catch (logError) {
      setJobLog(logError.message);
    }
  });

  useEffect(() => {
    refreshDashboard();
  }, [repoQuery]);

  useEffect(() => {
    if (!dashboard) {
      return undefined;
    }

    const interval = window.setInterval(
      () => refreshDashboard({ quiet: true }),
      activeJobs.length ? 5000 : 15000,
    );
    return () => window.clearInterval(interval);
  }, [dashboard?.generated_at, repoQuery, activeJobs.length]);

  useEffect(() => {
    refreshLog();
    if (!selectedJobId) {
      return undefined;
    }
    const interval = window.setInterval(() => refreshLog(), 4000);
    return () => window.clearInterval(interval);
  }, [selectedJobId, repoQuery]);

  useEffect(() => {
    if (!toast) {
      return undefined;
    }
    const timeout = window.setTimeout(() => setToast(null), 3600);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  useGSAP(
    () => {
      gsap.to(".floating-orb", {
        xPercent: 6,
        yPercent: -4,
        scale: 1.04,
        duration: 6,
        ease: "sine.inOut",
        yoyo: true,
        repeat: -1,
        stagger: 0.4,
      });

      gsap.fromTo(
        ".hero-card",
        { opacity: 0, y: 28 },
        {
          opacity: 1,
          y: 0,
          duration: 0.9,
          ease: "power3.out",
          stagger: 0.1,
        },
      );

      gsap.utils.toArray(".reveal-card").forEach((node) => {
        gsap.fromTo(
          node,
          { opacity: 0, y: 22 },
          {
            opacity: 1,
            y: 0,
            duration: 0.85,
            ease: "power3.out",
            scrollTrigger: {
              trigger: node,
              start: "top 90%",
            },
          },
        );
      });
    },
    { scope: rootRef, dependencies: [page, dashboard?.generated_at, selectedRunPath] },
  );

  async function runAction(work, successMessage) {
    setIsSubmitting(true);
    setError("");
    try {
      const response = await work();
      if (response?.job?.id) {
        setSelectedJobId(response.job.id);
      }
      setToast({ tone: "success", message: successMessage });
      startTransition(() => {
        setPage("ops");
      });
      await refreshDashboard({ quiet: true });
    } catch (actionError) {
      setError(actionError.message);
      setToast({ tone: "danger", message: actionError.message });
    } finally {
      setIsSubmitting(false);
    }
  }

  function applyRepoRoot() {
    setRepoQuery(repoDraft.trim());
  }

  function switchPage(nextPage) {
    startTransition(() => setPage(nextPage));
  }

  function filteredConfigs() {
    return (state.configs ?? []).filter((item) =>
      JSON.stringify(item).toLowerCase().includes(deferredSearch),
    );
  }

  function filteredScripts() {
    return (state.scripts ?? []).filter((item) =>
      JSON.stringify(item).toLowerCase().includes(deferredSearch),
    );
  }

  function filteredRuns() {
    return (state.runs ?? []).filter((item) =>
      JSON.stringify(item).toLowerCase().includes(deferredSearch),
    );
  }

  return (
    <div ref={rootRef} className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="floating-orb absolute left-[-10rem] top-[-6rem] h-80 w-80 rounded-full bg-[radial-gradient(circle,_rgba(87,230,255,0.22),_transparent_65%)] blur-3xl" />
        <div className="floating-orb absolute right-[-6rem] top-24 h-72 w-72 rounded-full bg-[radial-gradient(circle,_rgba(255,107,53,0.2),_transparent_68%)] blur-3xl" />
        <div className="floating-orb absolute bottom-[-10rem] left-1/3 h-96 w-96 rounded-full bg-[radial-gradient(circle,_rgba(128,255,177,0.14),_transparent_70%)] blur-3xl" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(129,152,189,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(129,152,189,0.06)_1px,transparent_1px)] bg-[size:72px_72px] [mask-image:radial-gradient(circle_at_center,black,transparent_92%)]" />
      </div>

      <div className="mx-auto flex max-w-[1500px] gap-6 px-4 py-5 lg:px-6">
        <aside className="hidden w-72 shrink-0 flex-col gap-5 lg:flex">
          <Panel className="hero-card sticky top-5 overflow-hidden">
            <div className="space-y-5">
              <div>
                <p className="eyebrow">Project Skull</p>
                <h1 className="font-display text-3xl uppercase tracking-[0.22em] text-white">
                  Command Deck
                </h1>
                <p className="mt-3 text-sm text-[var(--muted)]">
                  A sharper operations surface for training, evaluation, and artifact triage.
                </p>
              </div>

              <div className="space-y-2">
                {NAV_ITEMS.map((item) => (
                  <NavButton
                    key={item.key}
                    item={item}
                    active={page === item.key}
                    onClick={() => switchPage(item.key)}
                  />
                ))}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <MiniStat label="Active" value={activeJobs.length} tone="accent" />
                <MiniStat label="Runs" value={summary.run_count ?? 0} tone="info" />
                <MiniStat
                  label="Checkpoints"
                  value={summary.checkpoint_count ?? 0}
                  tone="success"
                />
                <MiniStat
                  label="Configs"
                  value={summary.config_count ?? 0}
                  tone="muted"
                />
              </div>
            </div>
          </Panel>
        </aside>

        <main className="flex-1 space-y-6">
          <Panel className="hero-card overflow-hidden">
            <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
              <div className="max-w-3xl space-y-4">
                <p className="eyebrow">Realtime control surface</p>
                <h2 className="font-display text-4xl uppercase tracking-[0.18em] text-white sm:text-5xl">
                  Train faster. See more. Panic less.
                </h2>
                <p className="max-w-2xl text-base leading-7 text-[var(--muted)]">
                  The old Streamlit control panel has been reimagined as a cinematic React dashboard
                  backed by the same Python job engine and repository scanner.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Badge tone="accent" icon={Sparkles}>
                    React + Tailwind + GSAP
                  </Badge>
                  <Badge tone="info" icon={Bot}>
                    Local-first orchestration
                  </Badge>
                  <Badge tone="success" icon={ShieldCheck}>
                    Same training and job flows
                  </Badge>
                </div>
              </div>

              <div className="w-full max-w-xl space-y-3">
                <label className="text-xs font-semibold uppercase tracking-[0.28em] text-[var(--muted)]">
                  Workspace root
                </label>
                <div className="flex flex-col gap-3 sm:flex-row">
                  <input
                    className="field-input flex-1"
                    value={repoDraft}
                    onChange={(event) => setRepoDraft(event.target.value)}
                    placeholder="Use default repo root"
                  />
                  <button className="action-button" onClick={applyRepoRoot} type="button">
                    Apply Root
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => refreshDashboard()}
                    type="button"
                  >
                    <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
                    Refresh
                  </button>
                </div>
                <div className="flex flex-wrap gap-3 text-xs text-[var(--muted)]">
                  <span>Connected root: {payload.repo_root || "loading..."}</span>
                  <span>Last sync: {formatDate(payload.generated_at)}</span>
                </div>
              </div>
            </div>
          </Panel>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={Gauge}
              title="Pipeline readiness"
              value={`${pipeline.filter((stage) => stage.status === "ready").length}/${pipeline.length}`}
              detail="Stages online"
              tone="accent"
            />
            <MetricCard
              icon={ChartNoAxesCombined}
              title="Training runs"
              value={formatNumber(summary.run_count ?? 0)}
              detail={`${formatNumber(summary.sample_count ?? 0)} samples captured`}
              tone="info"
            />
            <MetricCard
              icon={Database}
              title="Tokenizer models"
              value={formatNumber(summary.tokenizer_count ?? 0)}
              detail={`${formatNumber(summary.bin_directory_count ?? 0)} shard directories`}
              tone="success"
            />
            <MetricCard
              icon={Activity}
              title="Active jobs"
              value={formatNumber(activeJobs.length)}
              detail={
                activeJobs[0]
                  ? `${activeJobs[0].label} is ${activeJobs[0].status}`
                  : "No active job at the moment"
              }
              tone="muted"
            />
          </div>

          <div className="flex flex-wrap gap-3 lg:hidden">
            {NAV_ITEMS.map((item) => (
              <NavPill
                key={item.key}
                active={page === item.key}
                label={item.label}
                onClick={() => switchPage(item.key)}
              />
            ))}
          </div>

          {error ? (
            <Panel className="hero-card border-[rgba(255,109,122,0.35)]">
              <div className="flex items-start gap-3 text-[var(--danger)]">
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
                <div>
                  <p className="font-semibold text-white">Action blocked</p>
                  <p className="mt-1 text-sm text-[var(--danger)]">{error}</p>
                </div>
              </div>
            </Panel>
          ) : null}

          {page === "overview" ? (
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Status panorama"
                title="See the shape of the repo in one pass"
                detail="Pipeline health, module coverage, and recent activity are grouped into fast-glance cards."
              />

              <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Pipeline Pulse</h3>
                      <p className="panel-copy">Every stage the repo can currently execute.</p>
                    </div>
                    <Badge tone="info" icon={Activity}>
                      {pipeline.length} stages
                    </Badge>
                  </div>
                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    {pipeline.map((stage) => (
                      <div
                        key={stage.name}
                        className={`stage-card stage-${stage.status || "idle"}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="font-display text-sm uppercase tracking-[0.2em] text-white">
                              {stage.name}
                            </p>
                            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                              {stage.detail}
                            </p>
                          </div>
                          <StageBadge status={stage.status} />
                        </div>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Live Queue</h3>
                      <p className="panel-copy">The jobs that currently need your attention.</p>
                    </div>
                    <Badge tone={activeJobs.length ? "accent" : "success"} icon={Play}>
                      {activeJobs.length ? "Executing" : "Idle"}
                    </Badge>
                  </div>
                  <div className="mt-5 space-y-3">
                    {jobs.length ? (
                      jobs.slice(0, 5).map((job) => (
                        <div key={job.id} className="list-card">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="text-sm font-semibold text-white">{job.label}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                                {job.job_type}
                              </p>
                            </div>
                            <StageBadge status={job.status} />
                          </div>
                          <p className="mt-3 text-sm text-[var(--muted)]">
                            Started {formatDate(job.created_at)}
                          </p>
                        </div>
                      ))
                    ) : (
                      <EmptyState
                        icon={Activity}
                        title="No web jobs yet"
                        body="Launch train, eval, sample, or pytest tasks from the Launch view."
                      />
                    )}
                  </div>
                </Panel>
              </div>

              <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Module Density</h3>
                      <p className="panel-copy">A quick pulse on where code lives across the stack.</p>
                    </div>
                    <Badge tone="muted" icon={Blocks}>
                      Python surface
                    </Badge>
                  </div>
                  <div className="mt-5 space-y-3">
                    {Object.entries(summary.module_counts ?? {}).map(([name, count]) => (
                      <div key={name} className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-semibold capitalize text-white">{name}</span>
                          <span className="text-[var(--muted)]">{count} files</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5">
                          <div
                            className="h-2 rounded-full bg-[linear-gradient(90deg,var(--accent),var(--accent-soft))]"
                            style={{
                              width: `${Math.max(
                                12,
                                (Number(count) / Math.max(1, ...Object.values(summary.module_counts ?? {}))) *
                                  100,
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Recent Runs</h3>
                      <p className="panel-copy">Loss, checkpoints, and samples at a glance.</p>
                    </div>
                    <Badge tone="info" icon={BrainCircuit}>
                      {state.runs?.length ?? 0} runs
                    </Badge>
                  </div>
                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    {filteredRuns()
                      .slice(0, 4)
                      .map((run) => (
                        <button
                          key={run.relative_path}
                          type="button"
                          className={`run-card text-left ${selectedRun?.relative_path === run.relative_path ? "run-card-active" : ""}`}
                          onClick={() => {
                            setSelectedRunPath(run.relative_path);
                            switchPage("ops");
                          }}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-white">{run.name}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                                {run.kind}
                              </p>
                            </div>
                            <ChevronRight className="h-4 w-4 text-[var(--muted)]" />
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                            <InfoCell label="Step" value={formatNumber(run.latest_step ?? 0)} />
                            <InfoCell
                              label="Best val"
                              value={formatMaybe(run.best_val_loss)}
                            />
                          </div>
                          <div className="mt-4">
                            <Sparkline
                              rows={run.metrics_rows ?? []}
                              fields={["train_loss", "val_loss"]}
                            />
                          </div>
                        </button>
                      ))}
                  </div>
                </Panel>
              </div>
            </div>
          ) : null}

          {page === "launch" ? (
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Job launch studio"
                title="Kick off experiments without losing the details"
                detail="Each launcher keeps the original command behavior but wraps it in a more guided UI."
              />

              <div className="flex flex-wrap gap-3">
                {LAUNCH_MODES.map((item) => (
                  <NavPill
                    key={item.key}
                    label={item.label}
                    active={launchMode === item.key}
                    onClick={() => setLaunchMode(item.key)}
                  />
                ))}
              </div>

              {launchMode === "train" ? (
                <Panel className="reveal-card">
                  <div className="grid gap-8 xl:grid-cols-[0.78fr_1.22fr]">
                    <div className="space-y-4">
                      <h3 className="panel-title">Train Launcher</h3>
                      <p className="panel-copy">
                        Auto-detect pretraining versus CPT versus SFT, or pin the mode yourself.
                      </p>
                      <div className="info-stack">
                        <InfoCell label="Train configs" value={launchpad.train_configs.length} />
                        <InfoCell label="Active jobs" value={activeJobs.length} />
                        <InfoCell label="Repo root" value={payload.repo_root || "default"} />
                      </div>
                    </div>
                    <form
                      className="grid gap-4 md:grid-cols-2"
                      onSubmit={(event) => {
                        event.preventDefault();
                        runAction(
                          () =>
                            api.launchTrain({
                              repo_root: repoQuery,
                              config_path: trainForm.config_path,
                              requested_mode: trainForm.requested_mode,
                              use_accelerate: trainForm.use_accelerate,
                              num_processes: trainForm.use_accelerate
                                ? Number(trainForm.num_processes || 2)
                                : null,
                              label: trainForm.label,
                            }),
                          "Training job queued.",
                        );
                      }}
                    >
                      <Field label="Config">
                        <select
                          className="field-input"
                          value={trainForm.config_path}
                          onChange={(event) =>
                            setTrainForm((current) => ({
                              ...current,
                              config_path: event.target.value,
                            }))
                          }
                        >
                          {launchpad.train_configs.map((item) => (
                            <option key={item.path} value={item.path}>
                              {item.path}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Mode">
                        <select
                          className="field-input"
                          value={trainForm.requested_mode}
                          onChange={(event) =>
                            setTrainForm((current) => ({
                              ...current,
                              requested_mode: event.target.value,
                            }))
                          }
                        >
                          <option value="auto">auto</option>
                          <option value="pretrain">pretrain</option>
                          <option value="cpt">cpt</option>
                          <option value="sft">sft</option>
                        </select>
                      </Field>
                      <Field label="Job label">
                        <input
                          className="field-input"
                          value={trainForm.label}
                          onChange={(event) =>
                            setTrainForm((current) => ({
                              ...current,
                              label: event.target.value,
                            }))
                          }
                          placeholder="train:auto:run_name"
                        />
                      </Field>
                      <Field label="Processes">
                        <input
                          className="field-input"
                          value={trainForm.num_processes}
                          disabled={!trainForm.use_accelerate}
                          onChange={(event) =>
                            setTrainForm((current) => ({
                              ...current,
                              num_processes: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <label className="toggle-card md:col-span-2">
                        <input
                          checked={trainForm.use_accelerate}
                          onChange={(event) =>
                            setTrainForm((current) => ({
                              ...current,
                              use_accelerate: event.target.checked,
                            }))
                          }
                          type="checkbox"
                        />
                        <span>
                          <strong>Use Accelerate</strong>
                          <small>Launch through `accelerate.commands.launch`.</small>
                        </span>
                      </label>
                      <div className="md:col-span-2 flex flex-wrap gap-3">
                        <button className="action-button" disabled={isSubmitting} type="submit">
                          <Rocket className="h-4 w-4" />
                          Start Training
                        </button>
                        <button
                          className="ghost-button"
                          onClick={() => refreshDashboard()}
                          type="button"
                        >
                          <RefreshCw className="h-4 w-4" />
                          Refresh Inputs
                        </button>
                      </div>
                    </form>
                  </div>
                </Panel>
              ) : null}

              {launchMode === "eval" ? (
                <Panel className="reveal-card">
                  <div className="grid gap-8 xl:grid-cols-[0.78fr_1.22fr]">
                    <div className="space-y-4">
                      <h3 className="panel-title">Evaluation Launcher</h3>
                      <p className="panel-copy">
                        Pair an eval config with a checkpoint and stream JSON output into the job log.
                      </p>
                      <div className="info-stack">
                        <InfoCell label="Eval configs" value={launchpad.eval_configs.length} />
                        <InfoCell label="Checkpoints" value={launchpad.checkpoints.length} />
                      </div>
                    </div>
                    <form
                      className="grid gap-4 md:grid-cols-2"
                      onSubmit={(event) => {
                        event.preventDefault();
                        runAction(
                          () =>
                            api.launchEval({
                              repo_root: repoQuery,
                              ...evalForm,
                            }),
                          "Evaluation job queued.",
                        );
                      }}
                    >
                      <Field label="Eval config">
                        <select
                          className="field-input"
                          value={evalForm.config_path}
                          onChange={(event) =>
                            setEvalForm((current) => ({
                              ...current,
                              config_path: event.target.value,
                            }))
                          }
                        >
                          {launchpad.eval_configs.map((item) => (
                            <option key={item.path} value={item.path}>
                              {item.path}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Checkpoint">
                        <select
                          className="field-input"
                          value={evalForm.checkpoint_path}
                          onChange={(event) =>
                            setEvalForm((current) => ({
                              ...current,
                              checkpoint_path: event.target.value,
                            }))
                          }
                        >
                          {launchpad.checkpoints.map((path) => (
                            <option key={path} value={path}>
                              {path}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Job label">
                        <input
                          className="field-input"
                          value={evalForm.label}
                          onChange={(event) =>
                            setEvalForm((current) => ({
                              ...current,
                              label: event.target.value,
                            }))
                          }
                          placeholder="eval:best"
                        />
                      </Field>
                      <label className="toggle-card">
                        <input
                          checked={evalForm.print_json}
                          onChange={(event) =>
                            setEvalForm((current) => ({
                              ...current,
                              print_json: event.target.checked,
                            }))
                          }
                          type="checkbox"
                        />
                        <span>
                          <strong>Print JSON</strong>
                          <small>Append machine-readable results to the log.</small>
                        </span>
                      </label>
                      <div className="md:col-span-2">
                        <button className="action-button" disabled={isSubmitting} type="submit">
                          <Play className="h-4 w-4" />
                          Start Evaluation
                        </button>
                      </div>
                    </form>
                  </div>
                </Panel>
              ) : null}

              {launchMode === "sample" ? (
                <Panel className="reveal-card">
                  <div className="grid gap-8 xl:grid-cols-[0.78fr_1.22fr]">
                    <div className="space-y-4">
                      <h3 className="panel-title">Sample Launcher</h3>
                      <p className="panel-copy">
                        Generate qualitative text fast, with prompt control and decoding knobs.
                      </p>
                      <div className="info-stack">
                        <InfoCell label="Prompt seed" value="Hello from Project Skull" />
                        <InfoCell label="Default max tokens" value={sampleForm.max_new_tokens} />
                      </div>
                    </div>
                    <form
                      className="grid gap-4 md:grid-cols-2"
                      onSubmit={(event) => {
                        event.preventDefault();
                        runAction(
                          () =>
                            api.launchSample({
                              repo_root: repoQuery,
                              config_path: sampleForm.config_path,
                              checkpoint_path: sampleForm.checkpoint_path,
                              prompt: sampleForm.prompt,
                              max_new_tokens: Number(sampleForm.max_new_tokens),
                              temperature: Number(sampleForm.temperature),
                              top_k: sampleForm.top_k ? Number(sampleForm.top_k) : null,
                              label: sampleForm.label,
                            }),
                          "Sampling job queued.",
                        );
                      }}
                    >
                      <Field label="Config">
                        <select
                          className="field-input"
                          value={sampleForm.config_path}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              config_path: event.target.value,
                            }))
                          }
                        >
                          {launchpad.train_configs.map((item) => (
                            <option key={item.path} value={item.path}>
                              {item.path}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Checkpoint">
                        <select
                          className="field-input"
                          value={sampleForm.checkpoint_path}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              checkpoint_path: event.target.value,
                            }))
                          }
                        >
                          {launchpad.checkpoints.map((path) => (
                            <option key={path} value={path}>
                              {path}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field className="md:col-span-2" label="Prompt">
                        <textarea
                          className="field-input min-h-32 resize-y"
                          value={sampleForm.prompt}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              prompt: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <Field label="Max new tokens">
                        <input
                          className="field-input"
                          type="number"
                          value={sampleForm.max_new_tokens}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              max_new_tokens: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <Field label="Temperature">
                        <input
                          className="field-input"
                          type="number"
                          min="0.1"
                          step="0.1"
                          value={sampleForm.temperature}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              temperature: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <Field label="Top-k (optional)">
                        <input
                          className="field-input"
                          value={sampleForm.top_k}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              top_k: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <Field label="Job label">
                        <input
                          className="field-input"
                          value={sampleForm.label}
                          onChange={(event) =>
                            setSampleForm((current) => ({
                              ...current,
                              label: event.target.value,
                            }))
                          }
                          placeholder="sample:best"
                        />
                      </Field>
                      <div className="md:col-span-2">
                        <button className="action-button" disabled={isSubmitting} type="submit">
                          <Sparkles className="h-4 w-4" />
                          Generate Sample
                        </button>
                      </div>
                    </form>
                  </div>
                </Panel>
              ) : null}

              {launchMode === "test" ? (
                <Panel className="reveal-card">
                  <div className="grid gap-8 xl:grid-cols-[0.78fr_1.22fr]">
                    <div className="space-y-4">
                      <h3 className="panel-title">Pytest Launcher</h3>
                      <p className="panel-copy">
                        Fire off the full suite or target individual modules without leaving the app.
                      </p>
                      <div className="info-stack">
                        <InfoCell label="Detected tests" value={summary.test_count ?? 0} />
                        <InfoCell label="Default target" value="tests" />
                      </div>
                    </div>
                    <form
                      className="grid gap-4"
                      onSubmit={(event) => {
                        event.preventDefault();
                        runAction(
                          () =>
                            api.launchTest({
                              repo_root: repoQuery,
                              targets: splitLines(testForm.targets),
                              extra_args: splitArgs(testForm.extra_args),
                              label: testForm.label,
                            }),
                          "Test job queued.",
                        );
                      }}
                    >
                      <Field label="Targets">
                        <textarea
                          className="field-input min-h-28 resize-y"
                          value={testForm.targets}
                          onChange={(event) =>
                            setTestForm((current) => ({
                              ...current,
                              targets: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <Field label="Extra pytest args">
                        <input
                          className="field-input"
                          value={testForm.extra_args}
                          onChange={(event) =>
                            setTestForm((current) => ({
                              ...current,
                              extra_args: event.target.value,
                            }))
                          }
                          placeholder="-q tests/test_web_jobs.py"
                        />
                      </Field>
                      <Field label="Job label">
                        <input
                          className="field-input"
                          value={testForm.label}
                          onChange={(event) =>
                            setTestForm((current) => ({
                              ...current,
                              label: event.target.value,
                            }))
                          }
                        />
                      </Field>
                      <div>
                        <button className="action-button" disabled={isSubmitting} type="submit">
                          <TestTube2 className="h-4 w-4" />
                          Run Tests
                        </button>
                      </div>
                    </form>
                  </div>
                </Panel>
              ) : null}
            </div>
          ) : null}

          {page === "ops" ? (
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Operations view"
                title="Jobs and runs, separated cleanly"
                detail="Follow live logs on one side and drill into historical metrics on the other."
              />

              <div className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Job Queue</h3>
                      <p className="panel-copy">Select a job to inspect its command and live log tail.</p>
                    </div>
                    <Badge tone={activeJobs.length ? "accent" : "success"} icon={TerminalSquare}>
                      {activeJobs.length ? `${activeJobs.length} active` : "queue idle"}
                    </Badge>
                  </div>
                  <div className="mt-5 space-y-3">
                    {jobs.length ? (
                      jobs.map((job) => (
                        <button
                          key={job.id}
                          type="button"
                          className={`job-card text-left ${selectedJob?.id === job.id ? "job-card-active" : ""}`}
                          onClick={() => setSelectedJobId(job.id)}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-white">{job.label}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                                {job.job_type}
                              </p>
                            </div>
                            <StageBadge status={job.status} />
                          </div>
                          <p className="mt-3 text-xs text-[var(--muted)]">
                            {formatDate(job.created_at)}
                          </p>
                        </button>
                      ))
                    ) : (
                      <EmptyState
                        icon={TerminalSquare}
                        title="No jobs found"
                        body="Use the Launch studio to create your first tracked task."
                      />
                    )}
                  </div>
                </Panel>

                <Panel className="reveal-card">
                  <div className="flex flex-col gap-5">
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div>
                        <h3 className="panel-title">Job Console</h3>
                        <p className="panel-copy">
                          Command previews, return codes, and the latest log buffer.
                        </p>
                      </div>
                      {selectedJob ? <StageBadge status={selectedJob.status} /> : null}
                    </div>

                    {selectedJob ? (
                      <>
                        <div className="grid gap-3 md:grid-cols-4">
                          <InfoCell label="Started" value={formatDate(selectedJob.started_at)} />
                          <InfoCell label="Finished" value={formatDate(selectedJob.finished_at)} />
                          <InfoCell label="Return code" value={formatMaybe(selectedJob.returncode)} />
                          <InfoCell label="PID" value={selectedJob.child_pid || selectedJob.runner_pid || "n/a"} />
                        </div>

                        <div className="command-block">
                          <code>{selectedJob.display_command}</code>
                        </div>

                        <div className="flex flex-wrap items-center gap-3">
                          {isActiveStatus(selectedJob.status) ? (
                            <button
                              className="ghost-danger-button"
                              onClick={() =>
                                runAction(
                                  () => api.stopJob(selectedJob.id, repoQuery),
                                  `Stop requested for ${selectedJob.label}.`,
                                )
                              }
                              type="button"
                            >
                              <StopCircle className="h-4 w-4" />
                              Stop Job
                            </button>
                          ) : (
                            <>
                              <label className="toggle-card min-w-[220px]">
                                <input
                                  checked={deleteLogToo}
                                  onChange={(event) => setDeleteLogToo(event.target.checked)}
                                  type="checkbox"
                                />
                                <span>
                                  <strong>Delete log too</strong>
                                  <small>Remove the saved log file with the job record.</small>
                                </span>
                              </label>
                              <button
                                className="ghost-danger-button"
                                onClick={() =>
                                  runAction(
                                    () => api.deleteJob(selectedJob.id, repoQuery, deleteLogToo),
                                    `Deleted ${selectedJob.label}.`,
                                  )
                                }
                                type="button"
                              >
                                <Trash2 className="h-4 w-4" />
                                Delete Job
                              </button>
                            </>
                          )}
                        </div>

                        <div className="console-block">
                          <div className="console-header">
                            <span>log tail</span>
                            <button className="ghost-inline-button" onClick={() => refreshLog()} type="button">
                              <RefreshCw className="h-4 w-4" />
                              Refresh Log
                            </button>
                          </div>
                          <pre>{jobLog || "No log output yet."}</pre>
                        </div>
                      </>
                    ) : (
                      <EmptyState
                        icon={Activity}
                        title="Select a job"
                        body="Choose one from the queue to inspect live details."
                      />
                    )}
                  </div>
                </Panel>
              </div>

              <div className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Run Explorer</h3>
                      <p className="panel-copy">Filter by name, open the run, and inspect its artifacts.</p>
                    </div>
                    <div className="search-shell">
                      <Search className="h-4 w-4 text-[var(--muted)]" />
                      <input
                        className="search-input"
                        placeholder="Search runs, configs, scripts..."
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                      />
                    </div>
                  </div>
                  <div className="mt-5 space-y-3">
                    {filteredRuns().length ? (
                      filteredRuns().map((run) => (
                        <button
                          key={run.relative_path}
                          type="button"
                          className={`job-card text-left ${selectedRun?.relative_path === run.relative_path ? "job-card-active" : ""}`}
                          onClick={() => setSelectedRunPath(run.relative_path)}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-white">{run.name}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.24em] text-[var(--muted)]">
                                {run.relative_path}
                              </p>
                            </div>
                            <StageBadge status={run.best_val_loss != null ? "ready" : "partial"} />
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3">
                            <InfoCell label="Step" value={formatNumber(run.latest_step ?? 0)} />
                            <InfoCell label="Samples" value={formatNumber(run.sample_count ?? 0)} />
                          </div>
                        </button>
                      ))
                    ) : (
                      <EmptyState
                        icon={FolderKanban}
                        title="No matching runs"
                        body="Try a broader search or generate a run first."
                      />
                    )}
                  </div>
                </Panel>

                <Panel className="reveal-card">
                  {selectedRun ? (
                    <div className="space-y-5">
                      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                        <div>
                          <h3 className="panel-title">{selectedRun.name}</h3>
                          <p className="panel-copy">{selectedRun.relative_path}</p>
                        </div>
                        <Badge tone="info" icon={BrainCircuit}>
                          {selectedRun.kind}
                        </Badge>
                      </div>

                      <div className="grid gap-3 md:grid-cols-5">
                        <InfoCell label="Step" value={formatNumber(selectedRun.latest_step ?? 0)} />
                        <InfoCell
                          label="Train loss"
                          value={formatMaybe(selectedRun.latest_train?.train_loss)}
                        />
                        <InfoCell
                          label="Val loss"
                          value={formatMaybe(selectedRun.latest_val?.val_loss)}
                        />
                        <InfoCell label="Best val" value={formatMaybe(selectedRun.best_val_loss)} />
                        <InfoCell label="Checkpoints" value={selectedRun.checkpoint_count ?? 0} />
                      </div>

                      <Sparkline
                        rows={selectedRun.metrics_rows ?? []}
                        fields={["train_loss", "val_loss", "lr"]}
                        height={220}
                      />

                      <div className="flex flex-wrap gap-3">
                        {RUN_DETAIL_MODES.map((item) => (
                          <NavPill
                            key={item.key}
                            label={item.label}
                            active={runDetailMode === item.key}
                            onClick={() => setRunDetailMode(item.key)}
                          />
                        ))}
                      </div>

                      {runDetailMode === "checkpoints" ? (
                        <DataTable
                          columns={["name", "path", "size", "modified"]}
                          rows={(selectedRun.checkpoints ?? []).map((item) => ({
                            name: item.name,
                            path: item.relative_path,
                            size: item.size_human,
                            modified: item.modified_at,
                          }))}
                        />
                      ) : null}

                      {runDetailMode === "errors" ? (
                        <DataTable
                          columns={["step", "stage", "type", "message"]}
                          rows={(selectedRun.errors_rows ?? []).map((item) => ({
                            step: item.step ?? "n/a",
                            stage: item.stage ?? "n/a",
                            type: item.error_type ?? "n/a",
                            message: item.message ?? "n/a",
                          }))}
                        />
                      ) : null}

                      {runDetailMode === "samples" ? (
                        <div className="space-y-3">
                          {(selectedRun.samples ?? []).length ? (
                            selectedRun.samples.map((sample) => (
                              <div key={sample.relative_path} className="list-card">
                                <div className="flex items-start justify-between gap-3">
                                  <div>
                                    <p className="font-semibold text-white">{sample.name}</p>
                                    <p className="mt-1 text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                                      {sample.relative_path}
                                    </p>
                                  </div>
                                  <span className="text-xs text-[var(--muted)]">
                                    {formatDate(sample.modified_at)}
                                  </span>
                                </div>
                                <pre className="mt-4 whitespace-pre-wrap text-sm leading-6 text-[var(--muted)]">
                                  {sample.preview}
                                </pre>
                              </div>
                            ))
                          ) : (
                            <EmptyState
                              icon={Sparkles}
                              title="No saved samples"
                              body="Sampling outputs will appear here when the run emits text files."
                            />
                          )}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <EmptyState
                      icon={FolderKanban}
                      title="No run selected"
                      body="Once the repo has run artifacts, details will appear here."
                    />
                  )}
                </Panel>
              </div>
            </div>
          ) : null}

          {page === "assets" ? (
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Artifact registry"
                title="Inspect configs, data, and scripts without tab overload"
                detail="The browser now emphasizes dense but readable information blocks instead of raw tables everywhere."
              />

              <div className="flex flex-wrap gap-3">
                {ASSET_MODES.map((item) => (
                  <NavPill
                    key={item.key}
                    label={item.label}
                    active={assetMode === item.key}
                    onClick={() => setAssetMode(item.key)}
                  />
                ))}
              </div>

              {assetMode === "configs" ? (
                <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                  <Panel className="reveal-card">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="panel-title">Config Inventory</h3>
                        <p className="panel-copy">Train, model, and eval YAML discovered under `configs/`.</p>
                      </div>
                      <Badge tone="info" icon={FileCode2}>
                        {filteredConfigs().length} visible
                      </Badge>
                    </div>
                    <div className="mt-5">
                      <DataTable
                        columns={["path", "kind", "run", "device", "steps"]}
                        rows={filteredConfigs().map((item) => ({
                          path: item.relative_path,
                          kind: item.kind,
                          run: item.run_name ?? "n/a",
                          device: item.device ?? "n/a",
                          steps: item.max_steps ?? "n/a",
                        }))}
                      />
                    </div>
                  </Panel>
                  <Panel className="reveal-card">
                    <h3 className="panel-title">Corpus Registry</h3>
                    <p className="panel-copy">Mix definitions and source metadata from `configs/data/corpora.yaml`.</p>
                    <div className="mt-5 space-y-3">
                      {(state.corpora?.sources ?? []).map((source) => (
                        <div key={source.name} className="list-card">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-white">{source.name}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                                {source.domain} • {source.lang}
                              </p>
                            </div>
                            <Badge tone={source.enabled ? "success" : "muted"} icon={Database}>
                              {source.enabled ? "enabled" : "disabled"}
                            </Badge>
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3">
                            <InfoCell label="Tokenizer weight" value={source.tokenizer_sampling_weight} />
                            <InfoCell label="Pretrain weight" value={source.pretrain_weight} />
                            <InfoCell label="Clean path" value={source.clean_text_path || "n/a"} />
                            <InfoCell label="Bin output" value={source.bin_output_dir || "n/a"} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </Panel>
                </div>
              ) : null}

              {assetMode === "data" ? (
                <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
                  <AssetBucket
                    title="Tokenizers"
                    icon={BrainCircuit}
                    rows={state.data_assets?.tokenizers ?? []}
                  />
                  <AssetBucket
                    title="Clean corpora"
                    icon={Database}
                    rows={state.data_assets?.clean_files ?? []}
                  />
                  <AssetBucket
                    title="Bin shards"
                    icon={Boxes}
                    rows={state.data_assets?.bins ?? []}
                  />
                  <AssetBucket
                    title="Manifests"
                    icon={FolderKanban}
                    rows={state.data_assets?.manifests ?? []}
                  />
                </div>
              ) : null}

              {assetMode === "scripts" ? (
                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Script Rack</h3>
                      <p className="panel-copy">Automation helpers discovered in `scripts/`.</p>
                    </div>
                    <Badge tone="accent" icon={TerminalSquare}>
                      {filteredScripts().length} scripts
                    </Badge>
                  </div>
                  <div className="mt-5">
                    <DataTable
                      columns={["script", "args", "modified"]}
                      rows={filteredScripts().map((script) => ({
                        script: script.relative_path,
                        args: (script.arguments ?? []).join(" "),
                        modified: script.modified_at,
                      }))}
                    />
                  </div>
                </Panel>
              ) : null}
            </div>
          ) : null}

          {page === "guide" ? (
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Operator guide"
                title="Quick-start commands and the path through the pipeline"
                detail="This keeps the old help page useful, but makes it feel like part of the system instead of an afterthought."
              />

              <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
                <div className="grid gap-6">
                  {COMMAND_GROUPS.map((group) => (
                    <Panel key={group.title} className="reveal-card">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="panel-title">{group.title}</h3>
                          <p className="panel-copy">Ready to paste into your terminal.</p>
                        </div>
                        <Badge tone={group.tone} icon={TerminalSquare}>
                          command pack
                        </Badge>
                      </div>
                      <div className="command-block mt-5">
                        <code>{group.body}</code>
                      </div>
                    </Panel>
                  ))}
                </div>

                <Panel className="reveal-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="panel-title">Recommended Flow</h3>
                      <p className="panel-copy">A steady way to get from data to healthy runs.</p>
                    </div>
                    <Badge tone="success" icon={ShieldCheck}>
                      5-step path
                    </Badge>
                  </div>
                  <div className="mt-5 space-y-4">
                    {WORKFLOW_STEPS.map((step, index) => (
                      <div key={step} className="step-card">
                        <span className="step-index">{index + 1}</span>
                        <p className="text-sm leading-6 text-[var(--muted)]">{step}</p>
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </div>
          ) : null}
        </main>
      </div>

      {toast ? (
        <div className="pointer-events-none fixed bottom-5 right-5 z-50">
          <div className={`toast-shell toast-${toast.tone}`}>
            <p className="font-semibold text-white">{toast.message}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function NavButton({ item, active, onClick }) {
  const Icon = item.icon;
  return (
    <button
      className={`nav-button ${active ? "nav-button-active" : ""}`}
      onClick={onClick}
      type="button"
    >
      <span className="nav-icon">
        <Icon className="h-4 w-4" />
      </span>
      <span>{item.label}</span>
    </button>
  );
}

function NavPill({ active, label, onClick }) {
  return (
    <button
      className={`pill-button ${active ? "pill-button-active" : ""}`}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  );
}

function SectionHeader({ eyebrow, title, detail }) {
  return (
    <div className="reveal-card flex flex-col gap-3">
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="font-display text-2xl uppercase tracking-[0.18em] text-white sm:text-3xl">
        {title}
      </h2>
      <p className="max-w-3xl text-sm leading-7 text-[var(--muted)]">{detail}</p>
    </div>
  );
}

function Panel({ children, className = "" }) {
  return <section className={`surface-panel ${className}`}>{children}</section>;
}

function MetricCard({ icon: Icon, title, value, detail, tone }) {
  return (
    <Panel className="hero-card metric-card">
      <div className="flex items-start justify-between gap-3">
        <div className={`icon-shell icon-${tone}`}>
          <Icon className="h-5 w-5" />
        </div>
        <span className="text-xs uppercase tracking-[0.24em] text-[var(--muted)]">live</span>
      </div>
      <div className="mt-5">
        <p className="text-sm text-[var(--muted)]">{title}</p>
        <p className="mt-2 font-display text-4xl uppercase tracking-[0.18em] text-white">
          {value}
        </p>
        <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{detail}</p>
      </div>
    </Panel>
  );
}

function Badge({ tone, icon: Icon, children }) {
  return (
    <span className={`tone-badge tone-${tone}`}>
      {Icon ? <Icon className="h-3.5 w-3.5" /> : null}
      {children}
    </span>
  );
}

function StageBadge({ status }) {
  const tone = stageTone(status);
  return <span className={`stage-badge stage-badge-${tone}`}>{status || "idle"}</span>;
}

function MiniStat({ label, value, tone }) {
  return (
    <div className={`mini-stat mini-stat-${tone}`}>
      <span className="text-[0.68rem] uppercase tracking-[0.24em] text-[var(--muted)]">
        {label}
      </span>
      <strong className="mt-2 text-xl text-white">{value}</strong>
    </div>
  );
}

function Field({ children, className = "", label }) {
  return (
    <label className={`grid gap-2 ${className}`}>
      <span className="text-xs font-semibold uppercase tracking-[0.24em] text-[var(--muted)]">
        {label}
      </span>
      {children}
    </label>
  );
}

function InfoCell({ label, value }) {
  return (
    <div className="info-cell">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ body, icon: Icon, title }) {
  return (
    <div className="flex flex-col items-start gap-4 rounded-[1.25rem] border border-white/10 bg-white/[0.02] p-5">
      <div className="icon-shell icon-muted">
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="font-semibold text-white">{title}</p>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{body}</p>
      </div>
    </div>
  );
}

function DataTable({ columns, rows }) {
  if (!rows.length) {
    return (
      <div className="rounded-[1.25rem] border border-dashed border-white/12 px-4 py-8 text-center text-sm text-[var(--muted)]">
        Nothing to show yet.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-[1.25rem] border border-white/10">
      <div className="max-h-[28rem] overflow-auto">
        <table className="min-w-full divide-y divide-white/8 text-left text-sm">
          <thead className="sticky top-0 bg-[rgba(8,12,24,0.96)] backdrop-blur">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-4 py-3 text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-[var(--muted)]"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/6">
            {rows.map((row, index) => (
              <tr key={`${index}-${row[columns[0]]}`}>
                {columns.map((column) => (
                  <td key={column} className="px-4 py-3 align-top text-[var(--text-soft)]">
                    {String(row[column] ?? "n/a")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Sparkline({ rows, fields, height = 148 }) {
  const points = [];
  for (const row of rows.slice(-40)) {
    const value = fields.find((field) => typeof row[field] === "number");
    if (value) {
      points.push(Number(row[value]));
    }
  }

  if (points.length < 2) {
    return (
      <div className="grid h-[148px] place-items-center rounded-[1.25rem] border border-dashed border-white/10 text-sm text-[var(--muted)]">
        Metrics will render here once the run reports enough points.
      </div>
    );
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = Math.max(0.00001, max - min);
  const width = 420;
  const plot = points
    .map((point, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((point - min) / range) * (height - 14) - 7;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="rounded-[1.25rem] border border-white/8 bg-[rgba(255,255,255,0.02)] p-3">
      <svg className="h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="spark-gradient" x1="0%" x2="100%">
            <stop offset="0%" stopColor="var(--accent-soft)" />
            <stop offset="100%" stopColor="var(--signal)" />
          </linearGradient>
        </defs>
        <polyline
          fill="none"
          points={plot}
          stroke="url(#spark-gradient)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="4"
        />
      </svg>
    </div>
  );
}

function AssetBucket({ icon: Icon, rows, title }) {
  return (
    <Panel className="reveal-card">
      <div className="flex items-center justify-between">
        <div className="icon-shell icon-info">
          <Icon className="h-5 w-5" />
        </div>
        <Badge tone="muted" icon={Boxes}>
          {rows.length}
        </Badge>
      </div>
      <div className="mt-5">
        <h3 className="panel-title">{title}</h3>
        <p className="panel-copy">Most recent entries surfaced for quick inspection.</p>
      </div>
      <div className="mt-5 space-y-3">
        {rows.length ? (
          rows.slice(0, 5).map((row) => (
            <div key={row.relative_path} className="list-card">
              <p className="font-semibold text-white">{row.name}</p>
              <p className="mt-2 break-all text-sm leading-6 text-[var(--muted)]">
                {row.relative_path}
              </p>
            </div>
          ))
        ) : (
          <EmptyState
            icon={Icon}
            title={`No ${title.toLowerCase()}`}
            body="This bucket will populate as artifacts are created."
          />
        )}
      </div>
    </Panel>
  );
}

function stageTone(status) {
  if (["ready", "completed"].includes(status)) {
    return "success";
  }
  if (["running", "starting", "stop_requested", "stopping"].includes(status)) {
    return "accent";
  }
  if (["failed", "missing", "unknown"].includes(status)) {
    return "danger";
  }
  if (["partial", "stopped"].includes(status)) {
    return "warn";
  }
  return "muted";
}

function isActiveStatus(status) {
  return ["running", "starting", "stop_requested", "stopping"].includes(status);
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "0";
  }
  return new Intl.NumberFormat().format(Number(value));
}

function formatMaybe(value) {
  if (value == null || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Math.abs(value) >= 1000 ? formatNumber(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function formatDate(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString();
}

function splitLines(value) {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function splitArgs(value) {
  return value
    .split(/\s+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export default App;
