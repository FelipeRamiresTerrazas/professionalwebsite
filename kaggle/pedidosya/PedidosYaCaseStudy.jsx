/**
 * PedidosYaCaseStudy.jsx
 *
 * Product Operations Case Study — PedidosYa (Delivery Hero)
 * Scraped, cleaned, and analysed from 2,000 real Google Play reviews (BR).
 * All numbers are derived from the Python NLP pipeline (scrape_pedidosya.py
 * → analyze_friction.py).
 *
 * Assumes: React 17+ and Tailwind CSS available in host project.
 */

import React from "react";

// ─── Real data extracted by analyze_friction.py ────────────────────────────
const METRICS = {
  totalReviews: 2000,
  negativeReviews: 877,
  negativePct: 43.9,
  scoreDistribution: { "1": 779, "2": 98, "3": 79, "4": 177, "5": 867 },
  keywordFreq: {
    trava: 206,
    endereço: 75,
    busca: 9,
    carrinho: 2,
    filtro: 0,
  },
  topWords: [
    { word: "abre", count: 138 },
    { word: "funciona", count: 79 },
    { word: "ifood", count: 66 },
    { word: "tela", count: 52 },
    { word: "pagamento", count: 33 },
    { word: "entrega", count: 35 },
    { word: "péssimo", count: 47 },
    { word: "horrível", count: 32 },
  ],
  quotes: {
    trava: [
      "Aplicativo completamente disfuncional. Não dá pra mexer, trava e dá erro o tempo todo. Pouco intuitivo, sem opções óbvias pra ajudar na escolha.",
      "Oferece poucos cupons de desconto e trava com frequência.",
      "Pésima app, se trava todo el tiempo, 30 minutos para hacer un pedido.",
    ],
    endereço: [
      "Não marca a localização correta do envio.",
      "El app no reconoce mi dirección aunque esté escrita correctamente.",
      "GPS completamente errado, o endereço some na hora de confirmar.",
    ],
    pagamento: [
      "Siempre te cobran mal en la tarjeta, te cobran 2 veces y tardan en devolver.",
      "O carrinho bugou e cobrou duas vezes no cartão.",
      "O pagamento foi recusado sem motivo, mesmo com saldo disponível.",
    ],
    ifood: [
      "Desinstalei e fui pro iFood. Muito melhor.",
      "Como desenvolvedor mobile, é lamentável: o iFood está com um app muito mais enxuto.",
      "Prefiro mil vezes o iFood, esse app é uma bagunça.",
    ],
  },
};

// ─── Brand & UI constants ──────────────────────────────────────────────────
const RED = "#FF043C";
const RED_DARK = "#CC0030";
const DARK = "#0F172A";

// ─── Sub-components ────────────────────────────────────────────────────────

function HorizontalBar({ label, count, max, total, color = RED }) {
  const pct = Math.round((count / max) * 100);
  const ofNeg = total ? ((count / total) * 100).toFixed(1) : null;
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-slate-200">{label}</span>
        <span className="text-slate-400">
          {count} mentions
          {ofNeg && (
            <span className="text-white font-semibold ml-1">({ofNeg}%)</span>
          )}
        </span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2.5">
        <div
          className="h-2.5 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function ScoreBar({ stars, count, total }) {
  const pct = Math.round((count / total) * 100);
  const color =
    stars <= 2 ? RED : stars === 3 ? "#F59E0B" : "#22C55E";
  return (
    <div className="flex items-center gap-3 mb-2">
      <span className="text-slate-300 text-sm w-6 shrink-0">{stars}★</span>
      <div className="flex-1 bg-slate-700 rounded-full h-2">
        <div
          className="h-2 rounded-full"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-slate-400 text-sm w-14 text-right">
        {count} ({pct}%)
      </span>
    </div>
  );
}

function Quote({ text }) {
  return (
    <blockquote className="border-l-2 pl-3 py-1 italic text-slate-400 text-sm leading-relaxed"
      style={{ borderColor: RED }}>
      "{text}"
    </blockquote>
  );
}

function OpsCard({ icon, problem, impact, action, badgeColor }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-800/60 backdrop-blur-sm p-5 flex flex-col gap-3 hover:border-red-500/50 transition-colors">
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center text-xl shrink-0"
        style={{ backgroundColor: `${RED}22` }}
      >
        {icon}
      </div>
      <div>
        <span
          className="text-xs font-bold uppercase tracking-widest"
          style={{ color: RED }}
        >
          Problem Found
        </span>
        <p className="text-slate-200 text-sm mt-1 leading-relaxed">{problem}</p>
      </div>
      <div>
        <span className="text-xs font-bold uppercase tracking-widest text-amber-400">
          Impact Hypothesis
        </span>
        <p className="text-slate-300 text-sm mt-1 leading-relaxed">{impact}</p>
      </div>
      <div>
        <span className="text-xs font-bold uppercase tracking-widest text-emerald-400">
          Recommended Action
        </span>
        <p className="text-slate-300 text-sm mt-1 leading-relaxed">{action}</p>
      </div>
    </div>
  );
}

function PipelineStep({ num, icon, title, detail, arrow }) {
  return (
    <div className="flex items-start gap-2">
      <div className="flex flex-col items-center">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold text-white shrink-0"
          style={{ background: `linear-gradient(135deg, ${RED}, ${RED_DARK})` }}
        >
          {num}
        </div>
        {arrow && <div className="w-0.5 bg-slate-600 flex-1 mt-1" style={{ minHeight: 24 }} />}
      </div>
      <div className="pb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">{icon}</span>
          <span className="font-semibold text-white text-sm">{title}</span>
        </div>
        <p className="text-slate-400 text-xs leading-relaxed">{detail}</p>
      </div>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────

export default function PedidosYaCaseStudy() {
  const negKw = METRICS.negativeReviews;
  const maxKw = Math.max(...Object.values(METRICS.keywordFreq));

  return (
    <div
      className="min-h-screen font-sans antialiased"
      style={{ background: DARK, color: "#E2E8F0" }}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header
        className="sticky top-0 z-50 flex items-center gap-4 px-6 py-4 border-b border-slate-800"
        style={{ background: `${DARK}EE`, backdropFilter: "blur(12px)" }}
      >
        {/* PedidosYa wordmark pill */}
        <div
          className="px-3 py-1.5 rounded-lg text-white font-extrabold text-sm tracking-tight"
          style={{ background: RED }}
        >
          PedidosYa
        </div>
        <span className="text-slate-400 text-sm hidden sm:block">|</span>
        <span className="text-slate-300 text-sm hidden sm:block">
          Product Operations — App Friction Analysis
        </span>
        <div className="ml-auto flex gap-2">
          <span className="text-xs bg-slate-800 border border-slate-700 rounded-full px-3 py-1 text-slate-400">
            {METRICS.totalReviews.toLocaleString()} reviews
          </span>
          <span
            className="text-xs rounded-full px-3 py-1 font-semibold text-white"
            style={{ background: RED }}
          >
            {METRICS.negativePct}% negative
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-16">

        {/* ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────── */}
        <section id="executive-summary">
          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-1 rounded-full h-8"
              style={{ background: RED }}
            />
            <h2 className="text-2xl font-bold text-white">Executive Summary</h2>
          </div>
          <div
            className="rounded-2xl p-6 border border-slate-700 space-y-4"
            style={{ background: "rgba(255,4,60,0.06)" }}
          >
            <p className="text-slate-300 leading-relaxed">
              This study maps <strong className="text-white">hidden friction points</strong> in
              the PedidosYa (Delivery Hero) user journey by mining{" "}
              <strong className="text-white">2,000 real Google Play Store reviews</strong> from
              Brazilian users. The pipeline translates raw qualitative complaints into
              quantified Product Ops signals covering the{" "}
              <strong style={{ color: RED }}>Discovery</strong> and{" "}
              <strong style={{ color: RED }}>Checkout</strong> phases of the funnel.
            </p>
            <p className="text-slate-300 leading-relaxed">
              The findings reveal a{" "}
              <strong className="text-white">43.9% negative review rate</strong> — well above
              the industry benchmark of ~15–20% for mature delivery super-apps.
              The dominant pain cluster is <strong className="text-white">app stability</strong>:
              crashes, freezes, and blank screens block users before they ever reach a
              restaurant menu. A secondary cluster around{" "}
              <strong className="text-white">GPS/address failures</strong> and{" "}
              <strong className="text-white">payment errors</strong> completes a funnel where
              friction compounds at every stage, making user retention structurally fragile.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              {[
                { label: "Reviews Analyzed", value: "2,000", sub: "Google Play BR" },
                { label: "Negative Rate", value: "43.9%", sub: "1-2 star reviews" },
                { label: "Crash Mentions", value: "23.5%", sub: "of negative reviews" },
                { label: "iFood Churn Signal", value: "66", sub: "direct competitor refs" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-xl p-3 text-center border border-slate-700 bg-slate-800/50"
                >
                  <div className="text-2xl font-extrabold text-white">{m.value}</div>
                  <div className="text-xs text-slate-400 mt-1">{m.label}</div>
                  <div className="text-xs mt-0.5" style={{ color: RED }}>{m.sub}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── 2. FRICTION DASHBOARD ─────────────────────────────────────────── */}
        <section id="friction-dashboard">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 rounded-full h-8" style={{ background: RED }} />
            <h2 className="text-2xl font-bold text-white">Friction Dashboard</h2>
          </div>

          <div className="grid sm:grid-cols-2 gap-5">
            {/* Score distribution */}
            <div className="rounded-2xl p-5 border border-slate-700 bg-slate-800/60">
              <h3 className="font-semibold text-white mb-4">
                Rating Distribution
              </h3>
              {[5, 4, 3, 2, 1].map((s) => (
                <ScoreBar
                  key={s}
                  stars={s}
                  count={METRICS.scoreDistribution[String(s)]}
                  total={METRICS.totalReviews}
                />
              ))}
              <p className="text-xs text-slate-500 mt-3">
                Bimodal pattern: love/hate split typical of a reliability-degraded
                app with pockets of loyal users.
              </p>
            </div>

            {/* Keyword freq */}
            <div className="rounded-2xl p-5 border border-slate-700 bg-slate-800/60">
              <h3 className="font-semibold text-white mb-4">
                Friction Keyword Frequency
                <span className="text-xs text-slate-500 font-normal ml-2">
                  in 1-2★ reviews
                </span>
              </h3>
              {Object.entries(METRICS.keywordFreq)
                .sort((a, b) => b[1] - a[1])
                .map(([kw, cnt]) => (
                  <HorizontalBar
                    key={kw}
                    label={kw}
                    count={cnt}
                    max={maxKw}
                    total={negKw}
                  />
                ))}
            </div>

            {/* Top words heatmap-style */}
            <div className="rounded-2xl p-5 border border-slate-700 bg-slate-800/60 sm:col-span-2">
              <h3 className="font-semibold text-white mb-4">
                Top Organic Signals in Negative Reviews
              </h3>
              <div className="flex flex-wrap gap-2">
                {METRICS.topWords.map(({ word, count }) => {
                  const size = count >= 100 ? "text-base" : count >= 50 ? "text-sm" : "text-xs";
                  const opacity = count >= 100 ? 1 : count >= 50 ? 0.8 : 0.6;
                  return (
                    <span
                      key={word}
                      className={`px-3 py-1.5 rounded-lg font-semibold ${size}`}
                      style={{
                        background: `rgba(255,4,60,${opacity * 0.18})`,
                        border: `1px solid rgba(255,4,60,${opacity * 0.4})`,
                        color: `rgba(255,255,255,${opacity})`,
                      }}
                      title={`${count} mentions`}
                    >
                      {word}
                      <sup className="ml-1 font-normal text-slate-500">{count}</sup>
                    </span>
                  );
                })}
              </div>
              <p className="text-xs text-slate-500 mt-3">
                "abre" (138) and "funciona" (79) dominate — most negative reviews centre
                on the app simply not opening or working. "ifood" (66) is an active
                competitive churn signal.
              </p>
            </div>
          </div>

          {/* Real user quotes */}
          <div className="mt-5 grid sm:grid-cols-2 gap-5">
            {Object.entries(METRICS.quotes)
              .filter(([, qs]) => qs.length > 0)
              .map(([kw, qs]) => (
                <div
                  key={kw}
                  className="rounded-2xl p-5 border border-slate-700 bg-slate-800/40"
                >
                  <h4 className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: RED }}>
                    Real user quotes — {kw}
                  </h4>
                  <div className="space-y-3">
                    {qs.slice(0, 2).map((q, i) => <Quote key={i} text={q} />)}
                  </div>
                </div>
              ))}
          </div>
        </section>

        {/* ── 3. PRODUCT OPS LENS ───────────────────────────────────────────── */}
        <section id="product-ops-lens">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 rounded-full h-8" style={{ background: RED }} />
            <h2 className="text-2xl font-bold text-white">
              The &lsquo;Product Ops&rsquo; Lens
            </h2>
          </div>
          <p className="text-slate-400 text-sm mb-6">
            Four business recommendations derived directly from the quantified
            friction data above.
          </p>
          <div className="grid sm:grid-cols-2 gap-5">
            <OpsCard
              icon="💥"
              problem={`23.5% of all negative reviews (206 mentions) cite crashes, freezes, or the app failing to open. "abre" alone appears 138 times — before any purchase signal is even possible.`}
              impact="Users in the consideration phase hit a fatal error before reaching a restaurant menu. Zero impressions served, zero GMV, full churn — with zero recovery path."
              action="Enforce a Crash-Free Sessions SLA gate before any feature ships. Instrument Crashlytics alerts at ≥ 0.5% session crash rate. Establish a dedicated Platform Stability squad with veto power over the release train."
            />
            <OpsCard
              icon="📍"
              problem={`75 negative reviews (8.6%) reference GPS/address failures — the app cannot correctly detect where the user is or validate their delivery address.`}
              impact="Address failure is a first-mile block: the app cannot show relevant restaurants or estimate delivery fees, collapsing the entire discovery funnel for affected users."
              action="Integrate redundant geocoding fallback (Google Maps + HERE). Instrument 'address auto-confirm rate' as a leading funnel KPI. Ship a saved-addresses feature for returning users to bypass GPS dependency."
            />
            <OpsCard
              icon="💸"
              problem={`33 organic mentions of payment issues in negative reviews — double-charges, card declines, and refund failures at the highest-intent moment of the funnel.`}
              impact="A user who has browsed, selected items, and entered checkout has maximum purchase intent. Payment failure here yields 100% GMV loss from a warm, committed buyer."
              action="Audit payment gateway success rate by method (card vs cash vs PIX). Prioritise PIX as primary method (instant, low-failure). A/B test one-tap reorder to eliminate repeat checkout friction for loyal users."
            />
            <OpsCard
              icon="⚔️"
              problem={`iFood is explicitly named 66 times in negative reviews (7.5%) as the stated alternative — making it a direct, measurable competitive churn signal.`}
              impact="Each iFood mention represents a user who has already evaluated the alternative and published their decision. Word-of-mouth churn multiplies retention cost and poisons organic acquisition."
              action="Monitor iFood mention velocity as a leading churn KPI in weekly review dashboards. Launch a win-back flow targeting users dormant ≥ 21 days with a personalised discount. Benchmark UX patterns against iFood quarterly."
            />
          </div>
        </section>

        {/* ── 4. DATA ARCHITECTURE ─────────────────────────────────────────── */}
        <section id="data-architecture">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 rounded-full h-8" style={{ background: RED }} />
            <h2 className="text-2xl font-bold text-white">Data Architecture</h2>
          </div>
          <p className="text-slate-400 text-sm mb-6">
            End-to-end pipeline powering this case study — from raw Play Store
            data to actionable Product Ops insights.
          </p>
          <div
            className="rounded-2xl border border-slate-700 p-6"
            style={{ background: "rgba(255,4,60,0.04)" }}
          >
            <div className="grid sm:grid-cols-2 gap-x-8">
              <div>
                <PipelineStep
                  num="01"
                  icon="🕷️"
                  title="Python Scraper"
                  detail="google-play-scraper library targets com.pedidosya with lang=pt, country=br. Pagination via continuation_token collects 2,000 reviews in 200-row batches with polite rate limiting."
                  arrow
                />
                <PipelineStep
                  num="02"
                  icon="🧹"
                  title="Data Cleaning"
                  detail="Pandas pipeline: datetime normalisation, null-content filtering, score stratification. Negative subset isolated (1-2★: 877 rows)."
                  arrow
                />
                <PipelineStep
                  num="03"
                  icon="🔍"
                  title="NLP Keyword Analysis"
                  detail="Regex-based term frequency over PT+ES mixed corpus. Stops removal, Unicode-safe boundary matching. Counter-based top-25 organic signal extraction."
                  arrow
                />
              </div>
              <div>
                <PipelineStep
                  num="04"
                  icon="📊"
                  title="Metrics Serialisation"
                  detail="All frequencies, score distributions, and sampled quotes exported to pedidosya_friction_metrics.json — a clean contract between the Python pipeline and the frontend."
                  arrow
                />
                <PipelineStep
                  num="05"
                  icon="⚛️"
                  title="React Visualisation"
                  detail="Metrics embedded directly in component state. HorizontalBar, ScoreBar, OpsCard, PipelineStep sub-components render all data. Tailwind CSS for layout; brand tokens for PedidosYa identity."
                  arrow={false}
                />
              </div>
            </div>

            {/* Visual pipeline summary row */}
            <div className="mt-6 flex flex-wrap items-center gap-2 justify-center text-xs text-slate-400">
              {[
                "google-play-scraper",
                "→",
                "pandas / regex NLP",
                "→",
                "JSON metrics",
                "→",
                "React + Tailwind",
              ].map((s, i) => (
                s === "→" ? (
                  <span key={i} className="text-slate-600 font-bold">→</span>
                ) : (
                  <span
                    key={i}
                    className="px-2.5 py-1 rounded-md border border-slate-700 bg-slate-800"
                  >
                    {s}
                  </span>
                )
              ))}
            </div>
          </div>
        </section>

      </main>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800 mt-16 py-8 text-center text-slate-600 text-xs">
        <p>
          Built by{" "}
          <span className="text-slate-400 font-semibold">
            Felipe Ramires Terrazas
          </span>{" "}
          · Data scraped April 2026 · Python + React + Tailwind
        </p>
        <p className="mt-1">
          Not affiliated with Delivery Hero or PedidosYa. For portfolio purposes
          only.
        </p>
      </footer>
    </div>
  );
}
