"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

const AUTH_KEY = "cooledai_portal_auth";

type AuthContextType = {
  isAuthenticated: boolean;
  login: (username: string, password: string) => boolean;
  logout: () => void;
};

type AuthContextFull = AuthContextType & { hydrated: boolean };
const AuthContext = createContext<AuthContextFull | null>(null);

// Demo access: username "demo" / password "demo"
const DEMO_USER = "demo";
const DEMO_PASS = "demo";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsAuthenticated(sessionStorage.getItem(AUTH_KEY) === "true");
      setHydrated(true);
    }
  }, []);

  const login = (username: string, password: string): boolean => {
    if (username === DEMO_USER && password === DEMO_PASS) {
      sessionStorage.setItem(AUTH_KEY, "true");
      setIsAuthenticated(true);
      return true;
    }
    return false;
  };

  const logout = () => {
    sessionStorage.removeItem(AUTH_KEY);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, logout, hydrated }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
