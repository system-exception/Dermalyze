import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

/**
 * Top-level error boundary. Catches any unhandled runtime errors and
 * renders a friendly recovery screen instead of a blank page.
 */
class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error.message };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  handleReload = () => {
    window.location.href = '/';
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md w-full bg-white rounded-3xl border border-slate-200 shadow-sm p-10 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-red-50 rounded-2xl mb-6">
            <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 mb-2 tracking-tight">
            Something went wrong
          </h1>
          <p className="text-sm text-slate-500 mb-1">An unexpected error occurred in the application.</p>
          {this.state.message && (
            <p className="text-xs text-slate-400 font-mono bg-slate-50 rounded-lg px-3 py-2 mb-6 break-all">
              {this.state.message}
            </p>
          )}
          <button
            onClick={this.handleReload}
            className="w-full py-3 bg-teal-600 hover:bg-teal-700 text-white text-sm font-bold rounded-full transition-colors"
          >
            Reload Application
          </button>
        </div>
        <p className="mt-8 text-[11px] font-medium text-slate-400 uppercase tracking-widest">
          Dermalyze — Clinical Support Tool
        </p>
      </div>
    );
  }
}

export default ErrorBoundary;
