"use client";

import { useEffect, useRef } from "react";

// Approximate data center hub coordinates (lat, lng)
const DATA_CENTERS: [number, number][] = [
  [38.9, -77.0],   // Virginia, US
  [45.5, -122.6],  // Oregon, US
  [41.5, -93.6],   // Iowa, US
  [37.3, -121.9],  // Silicon Valley
  [51.5, -0.1],    // London
  [52.3, 4.9],     // Amsterdam
  [48.1, 11.6],    // Munich
  [1.3, 103.8],    // Singapore
  [35.6, 139.7],   // Tokyo
  [22.3, 114.2],   // Hong Kong
  [33.8, -118.2],  // Los Angeles
  [41.8, -87.6],   // Chicago
  [32.7, -97.0],   // Dallas
  [-33.9, 18.4],   // Cape Town
  [-33.8, 151.2],  // Sydney
];

export function GlobeDataCenters() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;
    const rotationSpeed = 0.00015; // Slow Earth-like spin

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const draw = () => {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) * 0.38;
      const aspectY = 0.42; // Slightly squashed for globe look

      const rot = time * rotationSpeed;

      // Clip to globe ellipse so graticule stays inside
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius, radius * aspectY, 0, 0, Math.PI * 2);
      ctx.clip();

      // Draw latitude lines (horizontal rings)
      const latStep = 30;
      for (let lat = -60; lat <= 60; lat += latStep) {
        const phi = (lat * Math.PI) / 180;
        const r = radius * Math.cos(phi);
        const y = radius * aspectY * Math.sin(phi);
        if (r > 2) {
          ctx.strokeStyle = "rgba(0, 255, 204, 0.12)";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.ellipse(cx, cy - y, r, r * aspectY, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Draw longitude lines (meridians) - curved arcs
      const lonStep = 30;
      for (let lon = 0; lon < 360; lon += lonStep) {
        const lambda = ((lon - 90) * Math.PI) / 180 + rot;
        ctx.strokeStyle = "rgba(0, 255, 204, 0.12)";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let lat = -90; lat <= 90; lat += 3) {
          const phi = (lat * Math.PI) / 180;
          const x = cx + radius * Math.cos(phi) * Math.cos(lambda);
          const y = cy - radius * aspectY * Math.sin(phi);
          if (lat === -90) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      ctx.restore();

      // Globe outline (stronger edge)
      ctx.strokeStyle = "rgba(0, 255, 204, 0.22)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius, radius * aspectY, 0, 0, Math.PI * 2);
      ctx.stroke();

      // Project data center points onto 2D (orthographic)
      const points: { x: number; y: number; lng: number }[] = [];
      for (const [lat, lng] of DATA_CENTERS) {
        const lngRot = ((lng * Math.PI) / 180 + rot) % (Math.PI * 2);
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = lngRot;
        const x = cx + radius * Math.sin(phi) * Math.cos(theta);
        const y = cy - radius * aspectY * Math.cos(phi);
        if (x >= cx - radius && x <= cx + radius) {
          points.push({ x, y, lng });
        }
      }

      // Interconnecting lines
      const lineOpacity = 0.08 + Math.sin(time * 0.002) * 0.04;
      ctx.strokeStyle = `rgba(0, 255, 204, ${lineOpacity})`;
      ctx.lineWidth = 0.8;

      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dist = Math.hypot(points[j].x - points[i].x, points[j].y - points[i].y);
          if (dist < radius * 1.8) {
            const dashPhase = (time * 0.05 + i * 0.5 + j * 0.3) % 20;
            ctx.setLineDash([4, 6]);
            ctx.lineDashOffset = -dashPhase;
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[j].x, points[j].y);
            ctx.stroke();
          }
        }
      }
      ctx.setLineDash([]);

      // Data center lights
      const pulse = 0.75 + Math.sin(time * 0.004) * 0.2;
      for (const p of points) {
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 12);
        gradient.addColorStop(0, `rgba(0, 255, 204, ${0.6 * pulse})`);
        gradient.addColorStop(0.5, `rgba(0, 255, 204, ${0.25 * pulse})`);
        gradient.addColorStop(1, "rgba(0, 255, 204, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(0, 255, 204, ${pulse})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      time += 16;
      animationId = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10 pointer-events-none"
      style={{ background: "#000000" }}
    />
  );
}
