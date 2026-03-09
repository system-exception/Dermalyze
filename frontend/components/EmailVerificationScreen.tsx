
import React from 'react';
import Button from './ui/Button';

interface EmailVerificationScreenProps {
  email: string;
  onNavigateToLogin: () => void;
  onResendEmail: () => Promise<void>;
}

const COOLDOWN_SECONDS = 60;

const EmailVerificationScreen: React.FC<EmailVerificationScreenProps> = ({
  email,
  onNavigateToLogin,
  onResendEmail,
}) => {
  const [resending,  setResending]  = React.useState(false);
  const [resendMsg,  setResendMsg]  = React.useState<string | null>(null);
  const [cooldown,   setCooldown]   = React.useState(0); // seconds remaining

  // Tick the cooldown down every second
  React.useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const handleResend = async () => {
    setResending(true);
    setResendMsg(null);
    try {
      await onResendEmail();
      setResendMsg('Verification email resent. Please check your inbox.');
      setCooldown(COOLDOWN_SECONDS); // start cooldown only on success
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to resend. Please try again.';
      setResendMsg(message);
      // no cooldown on failure — let the user retry
    } finally {
      setResending(false);
    }
  };

  const resendDisabled = resending || cooldown > 0;

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 bg-slate-50 min-h-screen">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-sm border border-slate-100 p-8 sm:p-10">
        <div className="text-center">
          {/* Mail icon */}
          <div className="inline-flex items-center justify-center w-20 h-20 bg-teal-50 rounded-full mb-6">
            <svg className="w-10 h-10 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
          </div>

          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mb-2">
            Verify Your Email
          </h1>

          <p className="text-slate-500 text-sm mb-2 leading-relaxed">
            We've sent a verification link to:
          </p>
          <p className="text-slate-800 font-semibold text-sm mb-6 break-all">
            {email || 'your email address'}
          </p>

          <div className="bg-slate-50 rounded-xl border border-slate-100 p-4 mb-6 text-left space-y-3">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Next Steps</p>
            <ol className="space-y-2 text-sm text-slate-600 list-decimal list-inside">
              <li>Open your email inbox</li>
              <li>Click the verification link in the email from Dermalyze</li>
              <li>Return here and log in to your account</li>
            </ol>
          </div>

          <div className="space-y-3">
            <Button onClick={onNavigateToLogin}>
              Go to Login
            </Button>

            <button
              onClick={handleResend}
              disabled={resendDisabled}
              className="w-full text-sm font-medium text-slate-500 hover:text-teal-600 transition-colors text-center py-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {resending
                ? 'Resending…'
                : cooldown > 0
                  ? `Resend available in ${cooldown}s`
                  : "Didn't receive it? Resend email"}
            </button>
          </div>

          {resendMsg && (
            <p className="mt-4 text-xs text-teal-600 font-medium">{resendMsg}</p>
          )}

          <div className="mt-6 p-3 bg-amber-50 border border-amber-100 rounded-lg">
            <p className="text-[11px] text-amber-700 leading-relaxed">
              <strong>Tip:</strong> If you don't see the email, check your spam or junk folder.
              The email may take a minute or two to arrive.
            </p>
          </div>
        </div>
      </div>

      <footer className="mt-12 text-center max-w-sm px-4">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed">
          Clinical Support Tool. Designed to assist medical professionals.
        </p>
      </footer>
    </div>
  );
};

export default EmailVerificationScreen;
