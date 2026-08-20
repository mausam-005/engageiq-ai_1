import React, { useEffect, useState } from "react";
import NudgePreferences from "./components/NudgePreferences";

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch("/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      {window.location.pathname === "/student/settings/nudge" ? (
        <NudgePreferences />
      ) : (
        <>
          <h1>EngageIQ AI</h1>
          <p>Agentic classroom engagement monitoring system</p>
          <p>
            Backend status: {" "}
            <strong>{health ? health.status : "checking..."}</strong>
          </p>
        </>
      )}
    </div>
  );
}
