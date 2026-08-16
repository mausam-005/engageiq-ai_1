import React, { useEffect, useState } from "react";

export default function NudgePreferences() {
  const [user, setUser] = useState(null);
  const [prefs, setPrefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const me = await fetch("/api/v1/users/me").then((r) => r.json());
        setUser(me);
        const p = await fetch(`/api/preferences/${me.id}`).then((r) => r.json());
        setPrefs(p);
      } catch (err) {
        setMessage({ type: "error", text: "Unable to load preferences" });
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !prefs) {
    return <div>Loading nudge preferences...</div>;
  }

  const setField = (k, v) => setPrefs((s) => ({ ...s, [k]: v }));

  async function save() {
    setSaving(true);
    setMessage(null);
    try {
      const res = await fetch(`/api/preferences/${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prefs),
      });
      if (!res.ok) throw new Error("save failed");
      const updated = await res.json();
      setPrefs(updated);
      setMessage({ type: "success", text: "Preferences saved" });
    } catch (err) {
      setMessage({ type: "error", text: "Failed to save preferences" });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div style={{ maxWidth: 700, margin: "1rem auto", padding: 16 }}>
      <h2>Nudge Preferences</h2>

      <section>
        <h3>Channels</h3>
        <label>
          <input
            type="checkbox"
            checked={prefs.notification_enabled}
            onChange={(e) => setField("notification_enabled", e.target.checked)}
          />
          {' '}Browser notification
        </label>
        <br />
        <label>
          <input
            type="checkbox"
            checked={prefs.overlay_enabled}
            onChange={(e) => setField("overlay_enabled", e.target.checked)}
          />
          {' '}Visual overlay
        </label>
        <br />
        <label>
          <input
            type="checkbox"
            checked={prefs.audio_enabled}
            onChange={(e) => setField("audio_enabled", e.target.checked)}
          />
          {' '}Audio
        </label>
      </section>

      <section style={{ marginTop: 12 }}>
        <h3>Quiet hours</h3>
        <div>
          <label>
            Start:{' '}
            <input
              type="time"
              value={prefs.quiet_hours_start || ""}
              onChange={(e) => setField("quiet_hours_start", e.target.value || null)}
            />
          </label>
        </div>
        <div>
          <label>
            End:{' '}
            <input
              type="time"
              value={prefs.quiet_hours_end || ""}
              onChange={(e) => setField("quiet_hours_end", e.target.value || null)}
            />
          </label>
        </div>
      </section>

      <section style={{ marginTop: 12 }}>
        <h3>Sensitivity</h3>
        <label>
          <input
            type="radio"
            name="sensitivity"
            value="less"
            checked={prefs.sensitivity === "less"}
            onChange={() => setField("sensitivity", "less")}
          />
          {' '}Less
        </label>
        <br />
        <label>
          <input
            type="radio"
            name="sensitivity"
            value="normal"
            checked={prefs.sensitivity === "normal"}
            onChange={() => setField("sensitivity", "normal")}
          />
          {' '}Normal
        </label>
        <br />
        <label>
          <input
            type="radio"
            name="sensitivity"
            value="more"
            checked={prefs.sensitivity === "more"}
            onChange={() => setField("sensitivity", "more")}
          />
          {' '}More
        </label>
      </section>

      <div style={{ marginTop: 16 }}>
        <button onClick={save} disabled={saving}>
          {saving ? "Saving..." : "Save preferences"}
        </button>
        {message && (
          <div style={{ marginTop: 8, color: message.type === "error" ? "#b91c1c" : "#15803d" }}>
            {message.text}
          </div>
        )}
      </div>
    </div>
  );
}
