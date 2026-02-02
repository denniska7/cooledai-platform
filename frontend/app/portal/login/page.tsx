"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";

export default function PortalLoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const { login } = useAuth();
  const router = useRouter();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (login(username, password)) {
      router.push("/portal/dashboard");
    } else {
      setError("Invalid credentials. Use demo / demo for access.");
    }
  };

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-medium tracking-tight text-white mb-2">
          CooledAI Portal
        </h1>
        <p className="text-sm text-white/50 mb-8">Access your thermal dashboard</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-transparent border border-white px-4 py-3 text-white placeholder-white/40 focus:border-[#00FFCC] focus:outline-none transition-colors"
            />
          </div>
          <div>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-transparent border border-white px-4 py-3 text-white placeholder-white/40 focus:border-[#00FFCC] focus:outline-none transition-colors"
            />
          </div>
          {error && (
            <p className="text-sm text-red-400">{error}</p>
          )}
          <button
            type="submit"
            className="w-full border border-white bg-white px-4 py-3 text-sm font-medium text-black transition-opacity hover:opacity-90"
          >
            Access
          </button>
        </form>

        <p className="mt-8 text-xs text-white/40 text-center">
          Demo: username <span className="text-white/60">demo</span> / password{" "}
          <span className="text-white/60">demo</span>
        </p>
      </div>
    </div>
  );
}
