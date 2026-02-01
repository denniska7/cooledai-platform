"""
CooledAI Universal Data Ingestor

Uses fuzzy column matching to identify and map columns from ANY log file
to normalized BaseNode attributes. Drop an EPYC CSV, HWiNFO export, or
vendor-specific log - no code changes required.

Column Mapping (fuzzy match):
- thermal_input: Tdie, Tctl, Temp, temperature, °C, CPU Temp
- power_draw: Power, Package Power, TDP, CPU Power, W
- cooling_output: Fan, RPM, FAN1, fan_speed, flow
- utilization: Utilization, Load, CPU Load, %
"""

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from typing import Optional, List, Tuple
from datetime import datetime

from core.hal.base_node import ServerNode, BaseNode

# Stale data threshold: data older than this triggers GUARD_MODE (Communication Timeout)
STALE_DATA_THRESHOLD_SECONDS = 5.0


# Fuzzy mapping: BaseNode attribute -> list of column name patterns (case-insensitive)
COLUMN_PATTERNS = {
    "thermal_input": [
        "tdie", "tctl", "temp", "temperature", "°c", "celsius",
        "cpu temp", "cpu (tctl/tdie)", "package temp", "core temp"
    ],
    "power_draw": [
        "power", "package power", "tdp", "cpu power", "watt", "w ",
        "consumption", "draw"
    ],
    "cooling_output": [
        "fan", "rpm", "fan1", "fan_speed", "flow", "speed",
        "fan (fan)", "cooling", "cfm"
    ],
    "utilization": [
        "utilization", "load", "cpu load", "%", "usage",
        "workload", "busy"
    ],
    "ambient_inlet_temp": [
        "inlet", "ambient", "room temp", "intake", "air temp",
        "ambient temp", "rack inlet", "cold aisle"
    ],
    "timestamp": [
        "timestamp", "time", "date", "datetime", "ts", "log time",
        "sample time", "capture time", "recorded"
    ],
}


class DataIngestor:
    """
    Universal data ingestor with fuzzy column matching.
    
    Automatically identifies columns in CSV/JSON and maps them to
    normalized BaseNode attributes. Supports multiple encodings and
    delimiters for maximum compatibility.
    """
    
    def __init__(self, node_type: type = ServerNode):
        """
        Initialize the ingestor.
        
        Args:
            node_type: BaseNode subclass to instantiate (default: ServerNode)
        """
        self.node_type = node_type
        self._column_map: dict = {}
    
    def _fuzzy_match_column(self, df: pd.DataFrame, attribute: str) -> Optional[str]:
        """
        Find column matching any pattern for the given attribute.
        Uses case-insensitive partial matching.
        
        Args:
            df: DataFrame with columns to search
            attribute: BaseNode attribute name (e.g., thermal_input)
            
        Returns:
            Column name if match found, else None
        """
        patterns = COLUMN_PATTERNS.get(attribute, [])
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    return col
        return None
    
    def _detect_columns(self, df: pd.DataFrame) -> dict:
        """
        Auto-detect column mapping using fuzzy matching.
        
        Returns:
            Dict mapping BaseNode attribute -> column name
        """
        col_map = {}
        for attr in COLUMN_PATTERNS.keys():
            col = self._fuzzy_match_column(df, attr)
            if col:
                col_map[attr] = col
        return col_map
    
    def _safe_numeric(self, series: pd.Series) -> np.ndarray:
        """Convert series to numeric, coercing errors to NaN, then fill."""
        s = pd.to_numeric(series, errors="coerce")
        s = s.ffill().bfill().fillna(0)
        return s.values.astype(np.float64)
    
    def _load_csv(self, source) -> pd.DataFrame:
        """
        Load CSV with flexible encoding and delimiter detection.
        Handles file paths, file-like objects, and Streamlit uploads.
        """
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
        delimiters = [",", ";", "\t"]
        
        # Get bytes
        if hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read()
            if isinstance(content, str):
                content = content.encode("utf-8", errors="replace")
        else:
            with open(source, "rb") as f:
                content = f.read()
        
        # Decode
        text = None
        for enc in encodings:
            try:
                text = content.decode(enc)
                break
            except (UnicodeDecodeError, AttributeError):
                continue
        if text is None:
            text = content.decode("utf-8", errors="replace")
        
        # Parse with delimiter detection
        read_kw = {"low_memory": False, "on_bad_lines": "skip"}
        for delim in delimiters:
            try:
                df = pd.read_csv(StringIO(text), sep=delim, **read_kw)
                if len(df.columns) > 1 and len(df) > 0:
                    return df
            except Exception:
                continue
        
        return pd.read_csv(StringIO(text), sep=",", **read_kw)
    
    def ingest_csv(
        self,
        source,
        column_map: Optional[dict] = None,
        node_id: str = "default",
    ) -> list:
        """
        Ingest CSV file and return list of BaseNode instances.
        
        Args:
            source: File path, file-like object, or bytes
            column_map: Optional override for auto-detected columns
            node_id: Node identifier for created nodes
            
        Returns:
            List of ServerNode (or configured node_type) instances
        """
        df = self._load_csv(source)
        
        # Use provided map or auto-detect
        self._column_map = column_map or self._detect_columns(df)
        
        # Require at least thermal_input for meaningful data
        if "thermal_input" not in self._column_map:
            # Fallback: first numeric column with values 15-95 (likely temp)
            for col in df.columns:
                try:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(vals) > 5:
                        med = vals.median()
                        if 15 < med < 95:
                            self._column_map["thermal_input"] = col
                            break
                except Exception:
                    continue
        
        nodes = []
        for idx, row in df.iterrows():
            node_data = {}
            ts = None
            for attr, col in self._column_map.items():
                if col in df.columns:
                    val = row[col]
                    if attr == "timestamp":
                        ts = self._parse_timestamp(val)
                    else:
                        try:
                            node_data[attr] = float(pd.to_numeric(val, errors="coerce") or 0)
                        except (TypeError, ValueError):
                            node_data[attr] = 0.0
            
            if ts is None:
                ts = datetime.now()
            node = self.node_type(
                node_id=node_id,
                timestamp=ts,
                source="csv",
                raw_data=row.to_dict(),
                **node_data,
            )
            nodes.append(node)
        
        return nodes
    
    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        column_map: Optional[dict] = None,
        node_id: str = "default",
    ) -> list:
        """
        Ingest pandas DataFrame directly.
        Same logic as ingest_csv but for in-memory DataFrames.
        """
        self._column_map = column_map or self._detect_columns(df)
        
        nodes = []
        for idx, row in df.iterrows():
            node_data = {}
            for attr, col in self._column_map.items():
                if col in df.columns:
                    val = row[col]
                    try:
                        node_data[attr] = float(pd.to_numeric(val, errors="coerce") or 0)
                    except (TypeError, ValueError):
                        node_data[attr] = 0.0
            
            node = self.node_type(
                node_id=node_id,
                timestamp=datetime.now(),
                source="dataframe",
                raw_data=row.to_dict(),
                **node_data,
            )
            nodes.append(node)
        
        return nodes
    
    def get_column_map(self) -> dict:
        """Return the detected/provided column mapping."""
        return self._column_map.copy()

    def _parse_timestamp(self, val) -> Optional[datetime]:
        """Parse timestamp from various formats (ISO, Unix, common strings)."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, datetime):
            return val
        try:
            s = str(val).strip()
            if not s:
                return None
            # Try pandas for flexible parsing
            ts = pd.to_datetime(s, errors="coerce")
            if pd.notna(ts):
                return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            # Unix timestamp
            f = float(s)
            if 1e9 < f < 2e9 or 1e12 < f < 2e12:  # Unix sec or ms
                return datetime.fromtimestamp(f / 1000 if f > 1e12 else f)
        except (ValueError, TypeError, OSError):
            pass
        return None

    def check_stale_data(
        self,
        nodes: List[BaseNode],
        max_age_seconds: float = STALE_DATA_THRESHOLD_SECONDS,
    ) -> Tuple[bool, str]:
        """
        Validate that node timestamps are not stale (older than max_age_seconds).
        If any node's data is older than threshold, triggers GUARD_MODE.

        Returns:
            (is_stale: bool, reason: str)
            If is_stale, reason contains 'Communication Timeout' alert text.
        """
        if not nodes:
            return False, ""
        now = datetime.now()
        stale_count = 0
        oldest_age_sec = 0.0
        for node in nodes:
            ts = getattr(node, "timestamp", None)
            if ts is None:
                continue
            try:
                age_sec = (now - ts).total_seconds()
                if age_sec > max_age_seconds:
                    stale_count += 1
                    oldest_age_sec = max(oldest_age_sec, age_sec)
            except (TypeError, AttributeError):
                pass
        if stale_count > 0:
            reason = (
                f"Communication Timeout: {stale_count}/{len(nodes)} nodes have data "
                f"older than {max_age_seconds:.0f}s (oldest: {oldest_age_sec:.1f}s). "
                "Stale data - trigger GUARD_MODE."
            )
            return True, reason
        return False, ""

    def compute_correlations(self, nodes: list) -> dict:
        """
        Multi-variate correlation: CPU Utilization vs Power Draw vs Ambient Inlet Temp.
        Returns correlation matrix and pairwise correlations.
        """
        if not nodes or len(nodes) < 3:
            return {"utilization_power": 0, "utilization_inlet": 0, "power_inlet": 0}

        util = np.array([getattr(n, "utilization", 0) or 0 for n in nodes])
        power = np.array([n.power_draw for n in nodes])
        inlet = np.array([
            getattr(n, "ambient_inlet_temp", None) or n.raw_data.get("ambient_inlet_temp") or 0
            for n in nodes
        ])

        def _corr(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        return {
            "utilization_power": _corr(util, power),
            "utilization_inlet": _corr(util, inlet),
            "power_inlet": _corr(power, inlet),
        }
