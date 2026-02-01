"""
CooledAI Watchdog Timer - Fail-Safe Thermal Control

The AI must 'kick the dog' (send a heartbeat) every N seconds.
If the AI hangs or crashes and the heartbeat is missed, the Watchdog
immediately triggers a Safe State command to hardware adapters.

Safe State bypasses the AI entirely and sets all cooling units to:
- Last known stable configuration, OR
- Default Fail-Safe High (100% cooling)

This prevents thermal runaway when the optimization software fails.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol
from datetime import datetime

# Configure watchdog logging
_logger = logging.getLogger("cooledai.watchdog")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


@dataclass
class SafeStateConfig:
    """
    Configuration for Safe State (Fail-Safe High).
    
    When Watchdog triggers, cooling units are set to this configuration.
    cooling_level: 1.0 = 100% (max cooling), 0.0 = off
    """
    cooling_level: float = 1.0   # Fail-Safe High = 100% cooling
    fan_rpm: Optional[float] = None   # Absolute RPM if known
    description: str = "Fail-Safe High (100% cooling)"


class CoolingControlProtocol(Protocol):
    """
    Protocol for adapters that can receive Safe State commands.
    
    Hardware adapters (Modbus, SNMP, etc.) implement set_safe_state()
    to apply cooling commands when Watchdog triggers.
    """
    
    def set_safe_state(self, config: SafeStateConfig) -> bool:
        """
        Apply Safe State configuration. Bypasses AI control.
        
        Args:
            config: SafeStateConfig with cooling_level (1.0 = 100%)
            
        Returns:
            True if command was applied successfully
        """
        ...


class Watchdog:
    """
    Watchdog Timer: AI must kick() every heartbeat_interval seconds.
    
    If heartbeat is missed (AI hung/crashed), immediately triggers
    Safe State on all registered cooling adapters. Safe State
    bypasses AI and sets cooling to Fail-Safe High (100%).
    """
    
    # Default: Fail-Safe High configuration
    DEFAULT_SAFE_STATE = SafeStateConfig(
        cooling_level=1.0,
        description="Fail-Safe High (100% cooling)",
    )
    
    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        check_interval: float = 1.0,
        safe_state: Optional[SafeStateConfig] = None,
    ):
        """
        Initialize Watchdog.
        
        Args:
            heartbeat_interval: Seconds between required kicks (default 10)
            check_interval: How often to check for timeout (default 1 sec)
            safe_state: Config to apply on timeout (default Fail-Safe High)
        """
        self.heartbeat_interval = heartbeat_interval
        self.check_interval = check_interval
        self.safe_state = safe_state or self.DEFAULT_SAFE_STATE
        
        # Last known stable config (updated when AI makes successful decisions)
        self._last_stable_config: Optional[SafeStateConfig] = None
        
        # Last heartbeat timestamp
        self._last_kick: Optional[float] = None
        self._lock = threading.Lock()
        
        # Registered cooling adapters (receive Safe State commands)
        self._adapters: List[CoolingControlProtocol] = []
        
        # Callback for custom Safe State handling (optional)
        self._safe_state_callback: Optional[Callable[[SafeStateConfig], None]] = None
        
        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # Track if Safe State was ever triggered (for alerting)
        self._safe_state_triggered = False
    
    def kick(self) -> None:
        """
        Kick the dog - AI must call this every cycle (within heartbeat_interval).
        
        Resets the timeout. If AI hangs and stops calling kick(), the
        Watchdog will trigger Safe State after heartbeat_interval seconds.
        """
        with self._lock:
            self._last_kick = time.monotonic()
    
    def register_adapter(self, adapter: CoolingControlProtocol) -> None:
        """
        Register a cooling adapter to receive Safe State commands.
        
        When Watchdog triggers, set_safe_state() is called on each
        registered adapter. Adapter must implement CoolingControlProtocol.
        """
        with self._lock:
            if adapter not in self._adapters:
                self._adapters.append(adapter)
                _logger.info("Watchdog: registered adapter %s", type(adapter).__name__)
    
    def unregister_adapter(self, adapter: CoolingControlProtocol) -> None:
        """Remove adapter from Safe State recipients."""
        with self._lock:
            if adapter in self._adapters:
                self._adapters.remove(adapter)
    
    def register_safe_state_callback(self, callback: Callable[[SafeStateConfig], None]) -> None:
        """
        Register callback for Safe State. Called when Watchdog triggers.
        Use for custom handling (e.g., API, control loop) that bypasses AI.
        """
        self._safe_state_callback = callback
    
    def update_last_stable(self, config: SafeStateConfig) -> None:
        """
        Update last known stable configuration.
        
        AI should call this when it successfully applies a cooling decision.
        On timeout, Watchdog will prefer last_stable over Fail-Safe High
        if available (optional behavior - currently uses Fail-Safe High
        for maximum safety).
        """
        with self._lock:
            self._last_stable_config = config
    
    def _trigger_safe_state(self) -> None:
        """
        Trigger Safe State: bypass AI, set all cooling to Fail-Safe High.
        
        Uses last known stable config if available and recent, otherwise
        uses default Fail-Safe High (100% cooling) for maximum safety.
        """
        # Prefer Fail-Safe High for maximum safety (user requirement)
        config = self.safe_state
        
        # Optional: use last stable if we have it (uncomment for that behavior)
        # if self._last_stable_config is not None:
        #     config = self._last_stable_config
        
        self._safe_state_triggered = True
        
        _logger.critical(
            "Watchdog: SAFE STATE TRIGGERED - AI heartbeat missed. "
            "Bypassing AI. Applying %s to all cooling units.",
            config.description,
        )
        
        # Notify registered adapters
        with self._lock:
            adapters = list(self._adapters)
        
        for adapter in adapters:
            try:
                success = adapter.set_safe_state(config)
                if success:
                    _logger.info("Watchdog: Safe State applied to %s", type(adapter).__name__)
                else:
                    _logger.warning("Watchdog: Failed to apply Safe State to %s", type(adapter).__name__)
            except Exception as e:
                _logger.error(
                    "Watchdog: Error applying Safe State to %s: %s",
                    type(adapter).__name__,
                    e,
                    exc_info=True,
                )
        
        # Call custom callback if registered
        if self._safe_state_callback:
            try:
                self._safe_state_callback(config)
            except Exception as e:
                _logger.error("Watchdog: Safe state callback failed: %s", e, exc_info=True)
    
    def _watch_loop(self) -> None:
        """Background thread: check for heartbeat timeout."""
        _logger.info(
            "Watchdog: Started. Heartbeat interval=%.1fs. AI must kick() every %.1fs.",
            self.heartbeat_interval,
            self.heartbeat_interval,
        )
        
        while not self._stop_event.wait(self.check_interval):
            with self._lock:
                last = self._last_kick
            
            now = time.monotonic()
            
            # If never kicked, wait for first kick before enforcing
            if last is None:
                continue
            
            elapsed = now - last
            if elapsed > self.heartbeat_interval:
                _logger.critical(
                    "Watchdog: Heartbeat missed! Last kick %.1fs ago (limit %.1fs). "
                    "Triggering Safe State.",
                    elapsed,
                    self.heartbeat_interval,
                )
                self._trigger_safe_state()
                # Reset kick so we don't spam - next kick will reset
                with self._lock:
                    self._last_kick = now
    
    def start(self) -> None:
        """Start the Watchdog background thread."""
        if self._running:
            _logger.warning("Watchdog: Already running")
            return
        
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        _logger.info("Watchdog: Running")
    
    def stop(self) -> None:
        """Stop the Watchdog background thread."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        _logger.info("Watchdog: Stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if Watchdog is running."""
        return self._running
    
    @property
    def safe_state_triggered(self) -> bool:
        """True if Safe State was ever triggered (for alerting)."""
        return self._safe_state_triggered
    
    def time_since_last_kick(self) -> Optional[float]:
        """Seconds since last kick, or None if never kicked."""
        with self._lock:
            if self._last_kick is None:
                return None
            return time.monotonic() - self._last_kick


class MockCoolingAdapter:
    """
    Mock adapter for testing. Implements CoolingControlProtocol.
    Use when no real hardware is connected.
    """
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.last_safe_state: Optional[SafeStateConfig] = None
    
    def set_safe_state(self, config: SafeStateConfig) -> bool:
        """Apply Safe State (mock - just stores config)."""
        self.last_safe_state = config
        _logger.info("MockAdapter %s: Safe State applied - cooling_level=%.2f", self.name, config.cooling_level)
        return True
