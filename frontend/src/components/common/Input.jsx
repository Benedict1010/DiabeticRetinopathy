export default function Input({ label, error, hint, className = '', ...props }) {
  return <label className={`input-group ${className}`}>
    <span>{label}</span>
    <input aria-invalid={Boolean(error)} {...props} />
    {hint && <small>{hint}</small>}{error && <small className="error-text">{error}</small>}
  </label>
}
