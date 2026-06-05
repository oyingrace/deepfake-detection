import { useRef, useState } from "react";

const ACCEPTED = ["video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/webm"];

export default function VideoUploader({ onAnalyse, isLoading }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }

  return (
    <div className="w-full flex flex-col items-center gap-10">
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => inputRef.current?.click()}
        className={`w-full aspect-[16/9] max-h-[320px] border-2 border-dashed rounded-3xl flex flex-col items-center justify-center cursor-pointer transition-all duration-200
          ${dragOver
            ? "border-indigo-400 bg-indigo-50/80 scale-[1.01]"
            : "border-slate-200 bg-white hover:border-slate-300 hover:shadow-md shadow-sm"}`}
      >
        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-5 transition-colors
          ${dragOver ? "bg-indigo-100" : "bg-slate-100"}`}>
          <svg className="w-7 h-7 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6h.1a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <p className="text-slate-600 font-medium">Drop video here</p>
        <p className="text-slate-400 text-sm mt-1">or click to browse</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {preview && (
        <video
          src={preview}
          controls
          className="w-full rounded-2xl border border-slate-200 bg-white shadow-sm"
          style={{ maxHeight: 280 }}
        />
      )}

      {file && (
        <button
          onClick={() => onAnalyse(file)}
          disabled={isLoading}
          className="px-10 py-3.5 rounded-full font-medium text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md"
        >
          {isLoading ? (
            <span className="flex items-center gap-2.5">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
              </svg>
              Analysing…
            </span>
          ) : "Analyse"}
        </button>
      )}
    </div>
  );
}
