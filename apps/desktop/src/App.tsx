/** App root — sidecar handshake, then Start ⇄ Hub.
 *
 * In development the sidecar is started manually:
 *   cd packages/core && python -m latos.server
 * The packaged app will spawn it as a Tauri sidecar (RB5+).
 */

import { useEffect, useState } from "react";
import { waitForCore } from "./lib/api";
import { Hub } from "./screens/Hub";
import { Start } from "./screens/Start";

type CoreStatus = "connecting" | "ready" | "down";
type Screen = "start" | "hub";

export default function App() {
  const [core, setCore] = useState<CoreStatus>("connecting");
  const [screen, setScreen] = useState<Screen>("start");

  useEffect(() => {
    waitForCore()
      .then(() => setCore("ready"))
      .catch(() => setCore("down"));
  }, []);

  if (core === "connecting") {
    return (
      <div className="flex h-full items-center justify-center text-secondary">
        Connecting to the Latos core…
      </div>
    );
  }

  if (core === "down") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-10 text-center">
        <p className="font-medium">The Latos core is not running.</p>
        <p className="text-sm text-secondary" data-selectable>
          Start it with: <code>python -m latos.server</code> (packages/core), then
          restart the app.
        </p>
      </div>
    );
  }

  return screen === "start" ? (
    <Start onProjectReady={() => setScreen("hub")} />
  ) : (
    <Hub onBack={() => setScreen("start")} />
  );
}
