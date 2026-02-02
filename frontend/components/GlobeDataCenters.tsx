"use client";

import { useEffect, useRef } from "react";
import { EARTH_LAND_POLYGONS } from "@/lib/earthLand";

// Data center coordinates (lat, lng)
const DATA_CENTERS: [number, number][] = [
  [38.9, -77.0], [45.5, -122.6], [41.5, -93.6], [37.3, -121.9],
  [51.5, -0.1], [52.3, 4.9], [48.1, 11.6], [1.3, 103.8],
  [35.6, 139.7], [22.3, 114.2], [33.8, -118.2], [41.8, -87.6],
  [32.7, -97.0], [-33.9, 18.4], [-33.8, 151.2],
];

function project(
  lat: number,
  lng: number,
  cx: number,
  cy: number,
  radius: number,
  rot: number
): { x: number; y: number; visible: boolean; shade: number } {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = ((lng * Math.PI) / 180 + rot) % (Math.PI * 2);
  const x = cx + radius * Math.sin(phi) * Math.cos(theta);
  const y = cy - radius * Math.cos(phi);
  const visible = Math.sin(phi) * Math.cos(theta) >= -0.02;
  const shade = Math.max(0, 0.5 + 0.5 * (Math.sin(phi) * Math.cos(theta) * 0.7 + Math.cos(phi) * 0.3));
  return { x, y, visible, shade };
}

export function GlobeDataCenters() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;
    const rotationSpeed = 0.00018;

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
      const size = Math.min(w, h);
      const radius = size * 0.32;

      const rot = time * rotationSpeed;

      // Atmospheric glow (soft halo around globe)
      const glowGrad = ctx.createRadialGradient(cx, cy, radius * 0.7, cx, cy, radius * 1.8);
      glowGrad.addColorStop(0, "rgba(100, 150, 220, 0.03)");
      glowGrad.addColorStop(0.6, "rgba(80, 130, 200, 0.02)");
      glowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = glowGrad;
      ctx.fillRect(0, 0, w, h);

      // Clip to globe circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.clip();

      // Ocean base - deep Earth blue
      ctx.fillStyle = "rgba(8, 35, 75, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Ocean gradient - lighter near lit edge
      const oceanGrad = ctx.createRadialGradient(
        cx - radius * 0.5, cy - radius * 0.5, 0,
        cx, cy, radius * 1.2
      );
      oceanGrad.addColorStop(0, "rgba(40, 90, 150, 0.25)");
      oceanGrad.addColorStop(0.5, "rgba(0, 0, 0, 0)");
      oceanGrad.addColorStop(1, "rgba(0, 10, 30, 0.3)");
      ctx.fillStyle = oceanGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw land masses - Natural Earth accurate coastlines
      for (const polygon of EARTH_LAND_POLYGONS) {
        const pts: { x: number; y: number; shade: number }[] = [];
        let anyVisible = false;
        for (const pt of polygon) {
          const [lat, lng] = pt as [number, number];
          const p = project(lat, lng, cx, cy, radius, rot);
          pts.push({ x: p.x, y: p.y, shade: p.shade });
          if (p.visible) anyVisible = true;
        }
        if (pts.length > 2 && anyVisible) {
          const avgShade = pts.reduce((s, p) => s + p.shade, 0) / pts.length;
          const gray = Math.round(105 + avgShade * 45);
          ctx.fillStyle = `rgba(${gray}, ${gray + 10}, ${gray + 15}, 0.94)`;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
          ctx.closePath();
          ctx.fill();

          // Subtle coastline outline for definition
          ctx.strokeStyle = `rgba(${gray - 15}, ${gray - 5}, ${gray}, 0.25)`;
          ctx.lineWidth = 0.4;
          ctx.stroke();
        }
      }

      // Latitude lines
      for (let lat = -60; lat <= 60; lat += 30) {
        const latRad = (lat * Math.PI) / 180;
        const r = radius * Math.cos(latRad);
        const y = radius * Math.sin(latRad);
        if (r > 3) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
          ctx.lineWidth = 0.4;
          ctx.beginPath();
          ctx.ellipse(cx, cy - y, r, r, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Longitude lines
      for (let lon = 0; lon < 360; lon += 30) {
        const lambda = ((lon - 90) * Math.PI) / 180 + rot;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
        ctx.lineWidth = 0.4;
        ctx.beginPath();
        for (let lat = -90; lat <= 90; lat += 4) {
          const phi = (90 - lat) * (Math.PI / 180);
          const x = cx + radius * Math.sin(phi) * Math.cos(lambda);
          const y = cy - radius * Math.cos(phi);
          if (lat === -90) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      ctx.restore();

      // Terminator (day/night line) - subtle
      ctx.save();
      ctx.globalAlpha = 0.15;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const termTheta = (Math.PI / 2) + rot;
      for (let lat = -90; lat <= 90; lat += 3) {
        const phi = (90 - lat) * (Math.PI / 180);
        const x = cx + radius * Math.sin(phi) * Math.cos(termTheta);
        const y = cy - radius * Math.cos(phi);
        if (lat === -90) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // Globe outline - crisp edge with subtle rim light
      ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Project data center points
      const points: { x: number; y: number }[] = [];
      for (const [lat, lng] of DATA_CENTERS) {
        const p = project(lat, lng, cx, cy, radius, rot);
        if (p.visible) points.push({ x: p.x, y: p.y });
      }

      // Interconnecting lines
      const lineOpacity = 0.14 + Math.sin(time * 0.002) * 0.06;
      ctx.strokeStyle = `rgba(255, 255, 255, ${lineOpacity})`;
      ctx.lineWidth = 0.8;

      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dist = Math.hypot(points[j].x - points[i].x, points[j].y - points[i].y);
          if (dist < radius * 1.6) {
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

      // Data center lights - white with glow
      const pulse = 0.88 + Math.sin(time * 0.003) * 0.1;
      for (const p of points) {
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 18);
        gradient.addColorStop(0, `rgba(255, 255, 255, ${0.92 * pulse})`);
        gradient.addColorStop(0.3, `rgba(255, 255, 255, ${0.45 * pulse})`);
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 18, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(255, 255, 255, ${pulse})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
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
