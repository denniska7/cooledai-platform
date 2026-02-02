"use client";

import { useEffect, useRef } from "react";

// Earth continent polygons [lat, lng] - Natural Earth–inspired outlines
const LAND_MASSES: [number, number][][] = [
  // North America (Alaska → Canada → USA → Mexico → Central America)
  [[72, -165], [70, -155], [69, -140], [60, -140], [55, -130], [49, -125], [45, -125], [40, -124], [38, -123], [34, -120], [32, -117], [28, -115], [25, -110], [22, -105], [20, -98], [25, -97], [30, -95], [32, -92], [30, -85], [25, -82], [28, -80], [35, -76], [42, -71], [45, -66], [48, -64], [55, -60], [60, -65], [65, -80], [70, -100], [72, -165]],
  // South America (Colombia → Brazil bulge → Argentina/Chile)
  [[12, -72], [8, -67], [2, -52], [-5, -52], [-12, -45], [-18, -40], [-25, -48], [-32, -55], [-38, -60], [-45, -65], [-52, -70], [-55, -72], [-54, -68], [-45, -65], [-35, -58], [-25, -50], [-15, -50], [-8, -55], [0, -52], [5, -60], [10, -68], [12, -72]],
  // Europe (Scandinavia → UK → Iberia → Mediterranean)
  [[71, -25], [70, -10], [68, 5], [65, 10], [62, 5], [60, 0], [55, -5], [52, -3], [50, 0], [48, 2], [45, 5], [43, 10], [42, 15], [44, 20], [47, 25], [50, 28], [55, 30], [60, 28], [63, 22], [65, 15], [68, 10], [71, -5], [71, -25]],
  // Africa (distinctive triangular shape)
  [[37, -6], [35, 0], [33, 10], [28, 15], [20, 10], [12, 5], [5, 10], [0, 15], [-8, 20], [-18, 28], [-28, 28], [-35, 22], [-35, 18], [-28, 18], [-15, 15], [-5, 12], [5, 5], [15, 0], [25, -5], [32, -8], [37, -6]],
  // Asia (Russia → China → India → Southeast Asia → Japan)
  [[72, 55], [70, 70], [65, 85], [55, 95], [45, 105], [40, 115], [35, 120], [30, 120], [28, 95], [25, 90], [22, 88], [20, 100], [18, 105], [15, 105], [12, 100], [25, 75], [35, 70], [45, 65], [55, 60], [65, 55], [70, 60], [72, 55]],
  // Australia
  [[-10, 115], [-12, 125], [-18, 128], [-25, 130], [-32, 135], [-36, 140], [-38, 148], [-36, 152], [-28, 153], [-25, 145], [-20, 140], [-15, 125], [-10, 115]],
  // Greenland
  [[83, -45], [78, -55], [72, -55], [68, -52], [65, -50], [62, -48], [65, -42], [72, -38], [78, -40], [83, -45]],
  // Japan
  [[45, 140], [43, 141], [38, 139], [35, 138], [33, 130], [34, 129], [36, 133], [40, 140], [45, 140]],
  // British Isles
  [[60, -7], [58, -6], [55, -6], [53, -4], [52, 0], [54, -2], [58, -5], [60, -7]],
  // Madagascar
  [[-12, 49], [-15, 48], [-20, 44], [-25, 44], [-25, 47], [-20, 48], [-15, 50], [-12, 49]],
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

      // Ocean base - Earth-like blue (Atlantic/Pacific)
      ctx.fillStyle = "rgba(15, 45, 90, 0.92)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // 3D sphere shading - light from top-right, shadow bottom-left (Earth-like)
      const lightX = cx - radius * 0.4;
      const lightY = cy - radius * 0.4;
      const shadeGrad = ctx.createRadialGradient(
        lightX, lightY, 0,
        cx, cy, radius * 1.3
      );
      shadeGrad.addColorStop(0, "rgba(80, 120, 160, 0.2)");
      shadeGrad.addColorStop(0.4, "rgba(0, 0, 0, 0)");
      shadeGrad.addColorStop(1, "rgba(0, 15, 40, 0.5)");
      ctx.fillStyle = shadeGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw land masses - Earth-like gray/tan with 3D shading
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
          const gray = Math.round(100 + avgShade * 50); // 100-150 range
          ctx.fillStyle = `rgba(${gray}, ${gray + 5}, ${gray + 8}, 0.92)`;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
          ctx.closePath();
          ctx.fill();
        }
      }

      // Latitude lines (subtle - Earth-like globe)
      for (let lat = -60; lat <= 60; lat += 30) {
        const latRad = (lat * Math.PI) / 180;
        const r = radius * Math.cos(latRad);
        const y = radius * Math.sin(latRad);
        if (r > 3) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.ellipse(cx, cy - y, r, r, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Longitude lines
      for (let lon = 0; lon < 360; lon += 30) {
        const lambda = ((lon - 90) * Math.PI) / 180 + rot;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 0.5;
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
