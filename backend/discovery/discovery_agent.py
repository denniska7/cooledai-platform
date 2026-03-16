"""
CooledAI DiscoveryAgent - Analyze raw Modbus/SNMP telemetry and suggest mapping.

Over a 10-minute window, collects unmapped telemetry, computes pattern summaries,
and uses an LLM to compare against known equipment profiles (Liebert, Stulz,
Schneider). Returns a Confidence Score and suggested Mapping Schema so users
don't have to manually label points during onboarding.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any

from backend.discovery.schemas import (
    RawTelemetryPoint,
    RawTelemetryWindow,
    MappingSchema,
    DiscoveryResult,
    NORMALIZED_ATTRIBUTES,
)
from backend.discovery.equipment_profiles import get_profiles_by_protocol, EquipmentProfile

logger = logging.getLogger(__name__)

# Default window length for discovery (seconds)
DISCOVERY_WINDOW_SECONDS = 600  # 10 minutes

# Minimum samples per point to consider for pattern analysis
MIN_SAMPLES_PER_POINT = 5


class DiscoveryAgent:
    """
    Analyzes raw, unmapped Modbus/SNMP telemetry over a configurable window and
    uses an LLM to suggest equipment type and point-to-attribute mapping.
    """

    def __init__(
        self,
        window_seconds: float = DISCOVERY_WINDOW_SECONDS,
        min_samples_per_point: int = MIN_SAMPLES_PER_POINT,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.window_seconds = window_seconds
        self.min_samples_per_point = min_samples_per_point
        self._openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._llm_model = llm_model or os.environ.get("COOLEDAI_LLM_MODEL", "gpt-4o-mini")
        self._points: Dict[str, RawTelemetryPoint] = {}
        self._window_start: Optional[float] = None

    def add_sample(
        self,
        protocol: str,
        point_id: str,
        address: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record one telemetry sample. point_id should be unique (e.g. modbus:40001, snmp:1.3.6.1.4.1.9...).
        """
        import time
        ts = timestamp if timestamp is not None else time.time()
        if point_id not in self._points:
            self._points[point_id] = RawTelemetryPoint(
                point_id=point_id,
                protocol=protocol,
                address=address,
                series=[],
            )
        self._points[point_id].add_sample(ts, value)
        if self._window_start is None:
            self._window_start = ts

    def ingest_window(self, window: RawTelemetryWindow) -> None:
        """Ingest a full window of raw telemetry (e.g. from a collector buffer)."""
        for pt in window.points:
            for ts, val in pt.series:
                self.add_sample(pt.protocol, pt.point_id, pt.address, val, ts)

    def clear(self) -> None:
        """Reset the agent state for a new discovery run."""
        self._points.clear()
        self._window_start = None

    def _build_pattern_summary(self) -> Dict[str, Any]:
        """Compute per-point statistics for LLM context."""
        summary = {"protocol": None, "points": [], "duration_seconds": 0.0}
        if not self._points:
            return summary
        end_ts = None
        for pid, pt in self._points.items():
            if len(pt.series) < self.min_samples_per_point:
                continue
            stats = pt.stats()
            if stats["count"] == 0:
                continue
            summary["points"].append({
                "point_id": pid,
                "protocol": pt.protocol,
                "address": pt.address,
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "sample_count": stats["count"],
            })
            if pt.series:
                times = [t for t, _ in pt.series]
                if end_ts is None:
                    summary["duration_seconds"] = max(times) - min(times) if len(times) > 1 else 0
                    end_ts = max(times)
                summary["protocol"] = pt.protocol
        return summary

    def _build_llm_prompt(self, pattern_summary: Dict[str, Any]) -> str:
        """Build prompt for LLM: patterns + equipment profiles, ask for JSON."""
        protocol = pattern_summary.get("protocol") or "modbus"
        profiles = get_profiles_by_protocol(protocol)
        profile_text = "\n\n".join(p.to_llm_description() for p in profiles)

        points_text = json.dumps(pattern_summary.get("points", []), indent=2)

        return f"""You are a data center equipment discovery assistant. Given raw, unmapped {protocol.upper()} telemetry over a 10-minute window, compare the observed value ranges and point addresses to the known equipment profiles below and suggest which vendor/profile best matches and how to map each point to normalized attributes.

Normalized attributes (use exactly these keys in mapping_schema): {", ".join(NORMALIZED_ATTRIBUTES)}

Known equipment profiles:
{profile_text}

Observed telemetry (point_id, address, min, max, mean, sample_count):
{points_text}

Respond with a single JSON object (no markdown, no code block) with exactly these keys:
- "suggested_vendor": string (e.g. "Liebert", "Stulz", "Schneider", "APC")
- "confidence_score": float between 0.0 and 1.0
- "mapping_schema": object mapping each observed point_id to one of the normalized attributes above, e.g. {{ "modbus:40001": "thermal_input", "modbus:40002": "cooling_output" }}
- "message": short human-readable explanation

Only include point_ids that appear in the observed telemetry. If confidence is low, still provide the best-guess mapping and set confidence_score below 0.6.
"""

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI or Anthropic, return parsed JSON."""
        if self._openai_key:
            return self._call_openai(prompt)
        if self._anthropic_key:
            return self._call_anthropic(prompt)
        logger.warning("DiscoveryAgent: No OPENAI_API_KEY or ANTHROPIC_API_KEY set; using rule-based fallback.")
        return None

    def _call_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            return self._parse_llm_json(text)
        except ImportError:
            logger.warning("openai package not installed; pip install openai")
            return None
        except Exception as e:
            logger.exception("OpenAI call failed: %s", e)
            return None

    def _call_anthropic(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self._anthropic_key)
            msg = client.messages.create(
                model=self._llm_model or "claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (msg.content[0].text if msg.content else "").strip()
            return self._parse_llm_json(text)
        except ImportError:
            logger.warning("anthropic package not installed; pip install anthropic")
            return None
        except Exception as e:
            logger.exception("Anthropic call failed: %s", e)
            return None

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response (strip markdown code blocks if present)."""
        text = text.strip()
        for pattern in (r"```(?:json)?\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"):
            m = re.search(pattern, text)
            if m:
                raw = m.group(1) if "```" in pattern else m.group(0)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("DiscoveryAgent: Could not parse LLM response as JSON")
            return None

    def _fallback_heuristic(self, pattern_summary: Dict[str, Any]) -> DiscoveryResult:
        """Rule-based suggestion when LLM is unavailable."""
        points = pattern_summary.get("points", [])
        protocol = pattern_summary.get("protocol") or "modbus"
        mapping = MappingSchema()
        # Heuristic: order points by mean value; assign temp (15-40), fan (0-3000), power (0-50k), util (0-100)
        attrs_by_typical_range = [
            ("thermal_input", 15.0, 45.0),
            ("cooling_output", 0.0, 3500.0),
            ("power_draw", 0.0, 50000.0),
            ("utilization", 0.0, 100.0),
        ]
        attr_used = set()
        for pt in sorted(points, key=lambda p: (p.get("mean") or 0)):
            mean = pt.get("mean")
            if mean is None:
                continue
            point_id = pt.get("point_id")
            if not point_id:
                continue
            for attr, low, high in attrs_by_typical_range:
                if attr in attr_used:
                    continue
                if low <= mean <= high:
                    mapping.point_to_attribute[point_id] = attr
                    attr_used.add(attr)
                    break
        confidence = 0.4 + 0.1 * len(mapping.point_to_attribute)  # 0.5-0.8
        return DiscoveryResult(
            confidence_score=min(1.0, confidence),
            suggested_vendor="Generic",
            suggested_mapping_schema=mapping,
            raw_metrics={"fallback_heuristic": True, "pattern_summary": pattern_summary},
            message="No LLM configured; mapping suggested by value-range heuristic. Set OPENAI_API_KEY for vendor-specific discovery.",
        )

    def discover(self) -> DiscoveryResult:
        """
        Run discovery on the current telemetry window. Returns Confidence Score
        and suggested Mapping Schema. Use add_sample() or ingest_window() first
        to fill the window (e.g. 10 minutes of data).
        """
        pattern_summary = self._build_pattern_summary()
        points = pattern_summary.get("points", [])
        if not points:
            return DiscoveryResult(
                confidence_score=0.0,
                suggested_vendor=None,
                suggested_mapping_schema=None,
                raw_metrics={"pattern_summary": pattern_summary},
                message="No telemetry points with enough samples. Collect at least 5 samples per point over the discovery window.",
            )

        prompt = self._build_llm_prompt(pattern_summary)
        parsed = self._call_llm(prompt)
        if not parsed:
            return self._fallback_heuristic(pattern_summary)

        # Build result from LLM response
        confidence = float(parsed.get("confidence_score", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        vendor = parsed.get("suggested_vendor")
        raw_map = parsed.get("mapping_schema") or {}
        mapping = MappingSchema(point_to_attribute=dict(raw_map))
        message = parsed.get("message") or ""

        return DiscoveryResult(
            confidence_score=confidence,
            suggested_vendor=vendor,
            suggested_mapping_schema=mapping,
            raw_metrics={
                "pattern_summary": pattern_summary,
                "llm_response_keys": list(parsed.keys()),
            },
            message=message,
        )


