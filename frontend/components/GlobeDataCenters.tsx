"use client";

import { useEffect, useRef } from "react";

// Simplified continent polygons [lat, lng] - recognizable Earth outlines
const LAND_MASSES: [number, number][][] = [
  // North America
  [[70, -140], [70, -100], [65, -80], [50, -125], [40, -125], [35, -120], [30, -115], [25, -100], [20, -100], [15, -90], [25, -80], [45, -65], [50, -60], [55, -65], [60, -140], [70, -140]],
  // South America
  [[12, -70], [10, -60], [0, -50], [-15, -50], [-25, -55], [-35, -70], [-50, -75], [-55, -70], [-40, -65], [-20, -60], [-5, -70], [5, -75], [12, -70]],
  // Europe
  [[71, -10], [65, 5], [55, 10], [50, 0], [45, -5], [43, 5], [45, 15], [50, 25], [55, 30], [60, 25], [65, 20], [71, -10]],
  // Africa
  [[37, -5], [35, 0], [32, 10], [15, 0], [5, 10], [-5, 15], [-18, 25], [-35, 20], [-35, 15], [-20, 15], [-5, 10], [10, 0], [25, -10], [37, -5]],
  // Asia
  [[70, 50], [65, 70], [55, 90], [50, 105], [40, 115], [30, 120], [25, 105], [30, 85], [35, 75], [45, 70], [55, 60], [65, 55], [70, 50]],
  // Australia
  [[-10, 115], [-15, 125], [-25, 130], [-35, 135], [-38, 145], [-35, 150], [-28, 115], [-20, 115], [-10, 115]],
  // Greenland
  [[83, -45], [78, -55], [72, -55], [65, -50], [60, -45], [65, -40], [75, -35], [83, -45]],
];

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
  // Shade: 1 = lit (top-right), 0 = shadow (bottom-left) for 3D depth
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
    const rotationSpeed = 0.00018; // Smooth rotation for 3D motion

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
      const radius = size * 0.32; // Perfect circle - same in both dimensions

      const rot = time * rotationSpeed;

      // Clip to globe circle (strictly circular)
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.clip();

      // Ocean base - dark blue-gray
      ctx.fillStyle = "rgba(15, 25, 40, 0.85)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // 3D sphere shading - light from top-right, shadow bottom-left
      const lightX = cx - radius * 0.4;
      const lightY = cy - radius * 0.4;
      const shadeGrad = ctx.createRadialGradient(
        lightX, lightY, 0,
        cx, cy, radius * 1.3
      );
      shadeGrad.addColorStop(0, "rgba(60, 70, 85, 0.25)");
      shadeGrad.addColorStop(0.4, "rgba(0, 0, 0, 0)");
      shadeGrad.addColorStop(1, "rgba(0, 0, 0, 0.55)");
      ctx.fillStyle = shadeGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw land masses - gray with 3D shading (lighter on lit side)
      for (const polygon of LAND_MASSES) {
        const pts: { x: number; y: number; shade: number }[] = [];
        let anyVisible = false;
        for (const [lat, lng] of polygon) {
          const p = project(lat, lng, cx, cy, radius, rot);
          pts.push({ x: p.x, y: p.y, shade: p.shade });
          if (p.visible) anyVisible = true;
        }
        if (pts.length > 2 && anyVisible) {
          const avgShade = pts.reduce((s, p) => s + p.shade, 0) / pts.length;
          const gray = Math.round(95 + avgShade * 55); // 95-150 range
          ctx.fillStyle = `rgba(${gray}, ${gray}, ${gray + 5}, 0.9)`;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
          ctx.closePath();
          ctx.fill();
        }
      }

      // Latitude lines (circular)
      for (let lat = -60; lat <= 60; lat += 30) {
        const latRad = (lat * Math.PI) / 180;
        const r = radius * Math.cos(latRad);
        const y = radius * Math.sin(latRad);
        if (r > 3) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.ellipse(cx, cy - y, r, r, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Longitude lines
      for (let lon = 0; lon < 360; lon += 30) {
        const lambda = ((lon - 90) * Math.PI) / 180 + rot;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
        ctx.lineWidth = 0.6;
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

      // Globe outline - crisp circular edge for 3D definition
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
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

      // Interconnecting lines (white)
      const lineOpacity = 0.12 + Math.sin(time * 0.002) * 0.05;
      ctx.strokeStyle = `rgba(255, 255, 255, ${lineOpacity})`;
      ctx.lineWidth = 0.7;

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

      // Data center lights - pure white (no green tint)
      const pulse = 0.85 + Math.sin(time * 0.003) * 0.12;
      for (const p of points) {
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 16);
        gradient.addColorStop(0, `rgba(255, 255, 255, ${0.9 * pulse})`);
        gradient.addColorStop(0.35, `rgba(255, 255, 255, ${0.4 * pulse})`);
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(255, 255, 255, ${pulse})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
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
