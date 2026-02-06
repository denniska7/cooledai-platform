import Link from "next/link";
import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0a0a0a] px-4 py-12">
      <Link href="/" className="mb-8">
        <img src="/logo.png" alt="CooledAI Logo" style={{ height: "80px", width: "auto" }} className="block" />
      </Link>
      <SignIn
        appearance={{
          variables: { colorPrimary: "#22c55e", colorBackground: "#0a0a0a" },
          elements: {
            rootBox: "mx-auto",
            card: "bg-[#141414] border border-white/10 shadow-xl",
          },
        }}
        afterSignInUrl="/portal"
        signUpUrl="/sign-up"
      />
    </div>
  );
}
