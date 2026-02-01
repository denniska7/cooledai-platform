import { BackendStatus } from "@/components/BackendStatus";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1">
        {/* Page content - add your existing content here */}
      </main>
      <footer className="border-t border-gray-200 py-4 px-6 flex items-center justify-center gap-4">
        <BackendStatus />
      </footer>
    </div>
  );
}
