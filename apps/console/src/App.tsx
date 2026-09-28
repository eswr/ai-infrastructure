import { Show, onMount } from "solid-js";
import { connect } from "./api";
import { connected } from "./state";
import FleetMap from "./components/FleetMap";
import RequestFlow from "./components/RequestFlow";
import LifecycleTimeline from "./components/LifecycleTimeline";
import DemoControls from "./components/DemoControls";
import MetricsBar from "./components/MetricsBar";

export default function App() {
  onMount(connect);

  return (
    <>
      <header>
        <h1>INFERENCE LAB — CONTROL PLANE CONSOLE</h1>
        <span class={`live ${connected() ? "" : "down"}`}>
          <span class="dot" />
          <span>{connected() ? "LIVE" : "DISCONNECTED"}</span>
        </span>
        <span class="scope-legend">
          <span class="badge meas">MEASURED — real Metal</span>
          <span class="badge sim">SIMULATED — control-plane only</span>
          <span class="badge mock">MOCK — failure injection</span>
        </span>
      </header>
      <main>
        <div class="col-left">
          <FleetMap />
        </div>
        <div class="col-right">
          <RequestFlow />
          <div class="row-bottom">
            <LifecycleTimeline />
            <DemoControls />
          </div>
        </div>
      </main>
      <MetricsBar />
    </>
  );
}
