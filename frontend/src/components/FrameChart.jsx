import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from "recharts";

export default function FrameChart({ scores }) {
  if (!scores || scores.length === 0) return null;

  const data = scores.map((score, i) => ({ seq: i + 1, score: Math.round(score * 100) }));

  return (
    <div className="w-full bg-white rounded-3xl border border-slate-100 p-10 shadow-sm">
      <p className="text-sm text-slate-400 mb-8 text-center">Sequence scores(How fake does each short clip look)</p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis
            dataKey="seq"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <Tooltip
            formatter={(v) => [`${v}%`, "Fake"]}
            contentStyle={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
              fontSize: 13,
            }}
            labelFormatter={(l) => `#${l}`}
          />
          <ReferenceLine y={50} stroke="#fca5a5" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#6366f1"
            strokeWidth={2.5}
            dot={(props) => {
              const { cx, cy, payload } = props;
              return (
                <circle
                  key={payload.seq}
                  cx={cx}
                  cy={cy}
                  r={5}
                  fill={payload.score >= 50 ? "#ef4444" : "#6366f1"}
                  stroke="#fff"
                  strokeWidth={2}
                />
              );
            }}
            activeDot={{ r: 7, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
