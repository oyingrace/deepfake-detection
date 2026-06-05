export default function ResultCard({ result }) {
  const isFake = result.label === "FAKE";
  const pct = Math.round(result.confidence * 100);
  const avg = result.frame_scores.length
    ? Math.round(result.frame_scores.reduce((a, b) => a + b, 0) / result.frame_scores.length * 100)
    : 0;

  return (
    <div className="w-full">
      <div
        className={`rounded-3xl p-12 md:p-16 text-center border shadow-sm
          ${isFake
            ? "bg-gradient-to-b from-red-50 to-white border-red-100"
            : "bg-gradient-to-b from-emerald-50 to-white border-emerald-100"}`}
      >
        <p className={`text-6xl md:text-7xl font-bold tracking-tight mb-3
          ${isFake ? "text-red-600" : "text-emerald-600"}`}>
          {result.label}
        </p>
        <p className="text-3xl font-semibold text-slate-700 tabular-nums">{pct}%</p>
        <p className="text-slate-400 text-sm mt-2">confidence</p>
      </div>

      <div className="grid grid-cols-2 gap-6 mt-8">
        <Stat label="Blink rate" value={`${result.blink_rate}/s`} />
        <Stat label="Sequences" value={String(result.frame_scores.length)} />
        <Stat label="Avg score" value={`${avg}%`} className="col-span-2 max-w-xs mx-auto w-full" />
      </div>
    </div>
  );
}

function Stat({ label, value, className = "" }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-100 px-8 py-6 text-center shadow-sm ${className}`}>
      <p className="text-2xl font-semibold text-slate-800 tabular-nums">{value}</p>
      <p className="text-slate-400 text-sm mt-1">{label}</p>
    </div>
  );
}
