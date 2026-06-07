import { useState } from "react";
import axios from "axios";
import { apiUrl } from "./api";
import VideoUploader from "./components/VideoUploader";
import ResultCard from "./components/ResultCard";
import FrameChart from "./components/FrameChart";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleAnalyse(file) {
    setError(null);
    setResult(null);
    setIsLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(apiUrl("/predict"), form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120_000,
      });
      setResult(data);
    } catch (e) {
      const status = e.response?.status;
      if (status === 404) {
        setError(
          "API not found (404). Redeploy the frontend after setting VITE_API_URL, or check vercel.json rewrites point to your Render URL."
        );
      } else {
        setError(e.response?.data?.detail || e.message || "Something went wrong.");
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 flex flex-col">
      <header className="px-8 py-8">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-xl font-semibold tracking-tight text-slate-800">
            Enhanced Deepfake Detector by <span className="text-emerald-600">ALINA CHIKWADO</span>
          </h1>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center px-8 pb-24">
        <div className="w-full max-w-3xl flex flex-col gap-16">
          <section className="flex flex-col items-center">
            <VideoUploader onAnalyse={handleAnalyse} isLoading={isLoading} />
          </section>

          {error && (
            <p className="text-center text-sm text-red-600 bg-red-50 border border-red-100 rounded-2xl py-4 px-6">
              {error}
            </p>
          )}

          {result && (
            <section className="flex flex-col gap-12">
              <ResultCard result={result} />
              <FrameChart scores={result.frame_scores} />
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
