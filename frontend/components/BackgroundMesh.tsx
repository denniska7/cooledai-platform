"use client";

import { useEffect, useRef } from "react";

export function BackgroundMesh() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const draw = () => {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;
      const gridSize = 60;
      const opacity = 0.03 + Math.sin(time * 0.002) * 0.01;

      for (let x = 0; x < w; x += gridSize) {
        for (let y = 0; y < h; y += gridSize) {
          const offset = Math.sin(time * 0.001 + x * 0.01 + y * 0.01) * 2;
          ctx.strokeStyle = `rgba(0, 255, 204, ${opacity})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(x + offset, y);
          ctx.lineTo(x + gridSize + offset, y);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x, y + offset);
          ctx.lineTo(x, y + gridSize + offset);
          ctx.stroke();
        }
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
