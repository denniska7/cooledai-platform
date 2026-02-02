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
    const rotationSpeed = 0.0003;

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
      const radius = Math.min(w, h) * 0.35;

      // Draw globe outline (ellipse)
      ctx.strokeStyle = "rgba(0, 255, 204, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius, radius * 0.4, 0, 0, Math.PI * 2);
      ctx.stroke();

      // Rotate longitude over time
      const rot = time * rotationSpeed;

      // Project data center points onto 2D (simple orthographic)
      const points: { x: number; y: number; lng: number }[] = [];
      for (const [lat, lng] of DATA_CENTERS) {
        const lngRot = ((lng * Math.PI) / 180 + rot) % (Math.PI * 2);
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = lngRot;
        const x = cx + radius * Math.sin(phi) * Math.cos(theta);
        const y = cy - radius * 0.4 * Math.cos(phi);
        if (x >= cx - radius && x <= cx + radius) {
          points.push({ x, y, lng });
        }
      }

      // Draw interconnecting lines (animate opacity)
      const lineOpacity = 0.03 + Math.sin(time * 0.002) * 0.02;
      ctx.strokeStyle = `rgba(0, 255, 204, ${lineOpacity})`;
      ctx.lineWidth = 0.5;

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

      // Draw data center lights
      const pulse = 0.6 + Math.sin(time * 0.004) * 0.2;
      for (const p of points) {
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 8);
        gradient.addColorStop(0, `rgba(0, 255, 204, ${0.4 * pulse})`);
        gradient.addColorStop(0.5, `rgba(0, 255, 204, ${0.15 * pulse})`);
        gradient.addColorStop(1, "rgba(0, 255, 204, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(0, 255, 204, ${0.9 * pulse})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
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
