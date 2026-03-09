
import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({ variant = 'primary', children, ...props }) => {
  const baseStyles = "w-full py-2.5 px-4 rounded-full font-medium transition-all duration-200 text-sm focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none";
  
  const variants = {
    primary: "bg-teal-600 text-white hover:bg-teal-700 focus:ring-teal-500 shadow-sm",
    secondary: "bg-transparent border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 focus:ring-slate-200"
  };

  return (
    <button {...props} className={`${baseStyles} ${variants[variant]}`}>
      {children}
    </button>
  );
};

export default Button;
