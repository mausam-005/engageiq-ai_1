import PropTypes from "prop-types";
import React, { useEffect, useState } from "react";

/**
 * Render a subtle, non-blocking nudge overlay that fades in and out.
 *
 * @param {{message: string, durationMs?: number}} props
 * @returns {JSX.Element|null}
 */
function NudgeOverlay({ message, durationMs = 5000 }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isMounted, setIsMounted] = useState(true);

  useEffect(() => {
    const showTimer = window.setTimeout(() => setIsVisible(true), 50);
    const hideTimer = window.setTimeout(() => setIsVisible(false), durationMs);
    const unmountTimer = window.setTimeout(() => setIsMounted(false), durationMs + 2000);

    return () => {
      window.clearTimeout(showTimer);
      window.clearTimeout(hideTimer);
      window.clearTimeout(unmountTimer);
    };
  }, [durationMs, message]);

  if (!isMounted) {
    return null;
  }

  return (
    <div
      aria-live="polite"
      role="status"
      style={{
        position: "fixed",
        right: "24px",
        bottom: "24px",
        maxWidth: "320px",
        padding: "12px 16px",
        borderRadius: "999px",
        background: "rgba(15, 23, 42, 0.8)",
        color: "#f9fafb",
        border: "1px solid rgba(255, 255, 255, 0.15)",
        boxShadow:
          "0 0 0 1px rgba(255, 255, 255, 0.08), 0 0 24px rgba(59, 130, 246, 0.25)",
        opacity: isVisible ? 1 : 0,
        transform: `translateY(${isVisible ? 0 : "8px"})`,
        transition: "opacity 2s ease, transform 2s ease",
        pointerEvents: "none",
        zIndex: 9999,
      }}
    >
      <span style={{ fontSize: "0.95rem" }}>{message}</span>
    </div>
  );
}

NudgeOverlay.propTypes = {
  message: PropTypes.string.isRequired,
  durationMs: PropTypes.number,
};

export default NudgeOverlay;
