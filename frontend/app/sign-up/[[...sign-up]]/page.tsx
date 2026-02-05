import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0a0a] px-4">
      <SignUp
        appearance={{
          variables: { colorPrimary: "#22c55e", colorBackground: "#0a0a0a" },
          elements: {
            rootBox: "mx-auto",
            card: "bg-[#141414] border border-white/10 shadow-xl",
          },
        }}
        afterSignUpUrl="/portal"
        signInUrl="/sign-in"
      />
    </div>
  );
}
