"""
Tests for core.hardware.xcc_fan_controller — XCC Redfish + IPMI LAN Fan Control

All tests mock urllib.request (Redfish) and subprocess (IPMI LAN)
so no actual BMC connections are made.
"""

from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure repo root is on sys.path
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from core.hardware.xcc_fan_controller import XCCFanController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_urlopen_response(status: int, body: dict) -> MagicMock:
    """Build a mock context-manager for urllib.request.urlopen."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _build_thermal_body(num_fans: int = 4, reading: int = 50) -> dict:
    return {
        "Fans": [
            {
                "MemberId": str(i),
                "Name": f"Fan {i+1}",
                "Reading": reading,
                "ReadingUnits": "Percent",
                "Status": {"Health": "OK"},
                "@odata.id": f"/redfish/v1/Chassis/1/Thermal/Fans/{i}",
            }
            for i in range(num_fans)
        ],
        "Temperatures": [],
    }


def _make_controller(working_method: str = "none", ipmi_lan: bool = False) -> XCCFanController:
    """Create a controller with mocked probe (already probed)."""
    ctrl = XCCFanController.__new__(XCCFanController)
    ctrl._host = "10.0.0.1"
    ctrl._user = "USERID"
    ctrl._pass = "***REDACTED_BMC_PASS***"
    ctrl._timeout = 1.0
    ctrl._available = True
    ctrl._manual_active = False
    ctrl._working_method = working_method
    ctrl._ipmi_lan_available = ipmi_lan
    ctrl._ipmi_lan_variant = ""
    ctrl._base_url = "https://10.0.0.1"
    cred = f"{ctrl._user}:{ctrl._pass}".encode()
    ctrl._auth = f"Basic {base64.b64encode(cred).decode()}"
    return ctrl


# ===========================================================================
# Probe Tests
# ===========================================================================

class TestXCCFanControllerProbe(unittest.TestCase):
    """XCCFanController.__init__ probe behaviour."""

    def test_no_host_disables(self):
        """No XCC_BMC_HOST → available=False immediately."""
        with patch.dict("os.environ", {}, clear=True):
            ctrl = XCCFanController(host="")
        self.assertFalse(ctrl.available)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_probe_unreachable_no_ipmi(self, mock_urlopen, mock_run):
        """BMC unreachable + no IPMI → not available."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        # IPMI also fails
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="timeout")
        ctrl = XCCFanController(host="10.0.0.1", timeout_s=0.1)
        self.assertFalse(ctrl.available)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_probe_success_thermal_patch(self, mock_urlopen, mock_run):
        """BMC reachable, Thermal PATCH returns 200 → working_method = redfish_thermal."""
        service_root = _make_urlopen_response(200, {"RedfishVersion": "1.6.0"})
        thermal_get = _make_urlopen_response(200, _build_thermal_body(4, 50))
        thermal_patch = _make_urlopen_response(200, {})

        mock_urlopen.side_effect = [service_root, thermal_get, thermal_patch]
        ctrl = XCCFanController(host="10.0.0.1")
        self.assertTrue(ctrl.available)
        self.assertEqual(ctrl._working_method, XCCFanController.METHOD_REDFISH_THERMAL)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_probe_thermal_405_tries_oem_action(self, mock_urlopen, mock_run):
        """Thermal PATCH returns 405, OEM action POST returns 200 → oem_action method."""
        import urllib.error
        service_root = _make_urlopen_response(200, {"RedfishVersion": "1.6.0"})
        thermal_get = _make_urlopen_response(200, _build_thermal_body(4, 50))
        thermal_patch_err = urllib.error.HTTPError(
            "url", 405, "Method Not Allowed", {}, None
        )
        oem_action_ok = _make_urlopen_response(200, {})
        oem_action_restore = _make_urlopen_response(200, {})

        mock_urlopen.side_effect = [
            service_root, thermal_get, thermal_patch_err,
            oem_action_ok, oem_action_restore,
        ]
        ctrl = XCCFanController(host="10.0.0.1")
        self.assertTrue(ctrl.available)
        self.assertEqual(ctrl._working_method, XCCFanController.METHOD_REDFISH_OEM_ACTION)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_probe_redfish_405_falls_to_ipmi_lan(self, mock_urlopen, mock_run):
        """All Redfish writes fail, IPMI LAN works → ipmi_lan method."""
        import urllib.error
        service_root = _make_urlopen_response(200, {"RedfishVersion": "1.6.0"})
        thermal_get = _make_urlopen_response(200, _build_thermal_body(3, 50))
        thermal_patch_405 = urllib.error.HTTPError("url", 405, "Not Allowed", {}, None)
        oem_action_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        oem_get_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        mock_urlopen.side_effect = [
            service_root, thermal_get, thermal_patch_405,
            oem_action_404,
            oem_get_404,
        ]
        # IPMI LAN chassis status succeeds
        mock_run.return_value = MagicMock(
            returncode=0, stdout="System Power         : on\nPower Overload       : false", stderr=""
        )
        ctrl = XCCFanController(host="10.0.0.1")
        self.assertTrue(ctrl.available)
        self.assertEqual(ctrl._working_method, XCCFanController.METHOD_IPMI_LAN)
        self.assertTrue(ctrl.ipmi_lan_available)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_probe_monitoring_only(self, mock_urlopen, mock_run):
        """Redfish reads work, writes fail, no IPMI → monitoring only."""
        import urllib.error
        service_root = _make_urlopen_response(200, {"RedfishVersion": "1.6.0"})
        thermal_get = _make_urlopen_response(200, _build_thermal_body(3, 50))
        thermal_patch_405 = urllib.error.HTTPError("url", 405, "Not Allowed", {}, None)
        oem_action_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        oem_get_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        mock_urlopen.side_effect = [
            service_root, thermal_get, thermal_patch_405,
            oem_action_404,
            oem_get_404,
        ]
        # IPMI LAN fails too
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no response")
        ctrl = XCCFanController(host="10.0.0.1")
        self.assertTrue(ctrl.available)  # monitoring-only is still "available"
        self.assertEqual(ctrl._working_method, XCCFanController.METHOD_NONE)


# ===========================================================================
# Redfish Fan Set Tests
# ===========================================================================

class TestXCCFanControllerSetRedfish(unittest.TestCase):
    """set_manual_fan_percent() Redfish paths."""

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_set_via_thermal_patch(self, mock_urlopen):
        """Thermal PATCH path: read fans then PATCH → True, manual_active=True."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_THERMAL)
        thermal_get = _make_urlopen_response(200, _build_thermal_body(4, 50))
        thermal_patch = _make_urlopen_response(200, {})
        mock_urlopen.side_effect = [thermal_get, thermal_patch]

        result = ctrl.set_manual_fan_percent(75)
        self.assertTrue(result)
        self.assertTrue(ctrl.manual_active)

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_set_via_oem_action(self, mock_urlopen):
        """OEM Action POST path → True."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_OEM_ACTION)
        mock_urlopen.side_effect = [_make_urlopen_response(200, {})]

        result = ctrl.set_manual_fan_percent(60)
        self.assertTrue(result)
        self.assertTrue(ctrl.manual_active)

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_set_via_oem(self, mock_urlopen):
        """OEM PATCH path → True."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_OEM)
        oem_patch = _make_urlopen_response(200, {})
        mock_urlopen.side_effect = [oem_patch]

        result = ctrl.set_manual_fan_percent(60)
        self.assertTrue(result)
        self.assertTrue(ctrl.manual_active)

    def test_set_not_available(self):
        """Not available → returns False without any network calls."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_THERMAL)
        ctrl._available = False
        result = ctrl.set_manual_fan_percent(50)
        self.assertFalse(result)

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_set_clamps_pct(self, mock_urlopen):
        """Percentage is clamped to 0-100."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_OEM)
        oem_patch = _make_urlopen_response(200, {})
        mock_urlopen.side_effect = [oem_patch]

        result = ctrl.set_manual_fan_percent(150)  # should clamp to 100
        self.assertTrue(result)
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["FanSpeedPercent"], 100)


# ===========================================================================
# IPMI over LAN Fan Set Tests
# ===========================================================================

class TestXCCFanControllerIPMILan(unittest.TestCase):
    """IPMI over LAN fan control paths."""

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_set_via_ipmi_lan_lenovo(self, mock_run):
        """Lenovo OEM 0x3a succeeds → method cached as lenovo_0x3a."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)

        # Both calls succeed (manual mode + set speed)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = ctrl.set_manual_fan_percent(60)
        self.assertTrue(result)
        self.assertTrue(ctrl.manual_active)
        self.assertEqual(ctrl._ipmi_lan_variant, "lenovo_0x3a")

        # Verify ipmitool was called with lanplus
        calls = mock_run.call_args_list
        self.assertTrue(len(calls) >= 2)
        first_cmd = calls[0][0][0]
        self.assertIn("lanplus", first_cmd)
        self.assertIn("-H", first_cmd)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_set_via_ipmi_lan_dell_fallback(self, mock_run):
        """Lenovo 0x3a fails, Dell 0x30 succeeds → variant=dell_0x30."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)

        # Lenovo fails (returncode=1), Dell succeeds (returncode=0)
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="Invalid command"),  # lenovo manual mode
            MagicMock(returncode=0, stdout="", stderr=""),  # dell manual mode
            MagicMock(returncode=0, stdout="", stderr=""),  # dell set speed
        ]

        result = ctrl.set_manual_fan_percent(50)
        self.assertTrue(result)
        self.assertEqual(ctrl._ipmi_lan_variant, "dell_0x30")

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_ipmi_lan_both_fail(self, mock_run):
        """Both Lenovo and Dell IPMI commands fail → returns False."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        result = ctrl.set_manual_fan_percent(50)
        self.assertFalse(result)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_ipmi_lan_ipmitool_not_found(self, mock_run):
        """ipmitool not installed → graceful failure."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        mock_run.side_effect = FileNotFoundError("ipmitool")

        result = ctrl.set_manual_fan_percent(50)
        self.assertFalse(result)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_ipmi_lan_timeout(self, mock_run):
        """ipmitool times out → graceful failure."""
        import subprocess as sp
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        mock_run.side_effect = sp.TimeoutExpired(cmd="ipmitool", timeout=3)

        result = ctrl.set_manual_fan_percent(50)
        self.assertFalse(result)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_cached_variant_used(self, mock_run):
        """Once variant is cached, only that variant's commands are used."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        ctrl._ipmi_lan_variant = "lenovo_0x3a"

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = ctrl.set_manual_fan_percent(70)
        self.assertTrue(result)

        # Should only have called Lenovo commands (manual + set), not Dell
        calls = mock_run.call_args_list
        for c in calls:
            cmd = c[0][0]
            self.assertNotIn("0x30", cmd)  # Dell commands not used


# ===========================================================================
# Fallback (Redfish fails, IPMI LAN takes over)
# ===========================================================================

class TestXCCFanControllerFallback(unittest.TestCase):
    """When cached Redfish method fails, IPMI LAN picks up."""

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_redfish_fail_ipmi_lan_succeeds(self, mock_urlopen, mock_run):
        """Redfish PATCH 405 → falls through to IPMI LAN → success."""
        import urllib.error
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_THERMAL, ipmi_lan=True)

        # Thermal GET ok, PATCH fails
        thermal_get = _make_urlopen_response(200, _build_thermal_body(3, 50))
        thermal_patch_405 = urllib.error.HTTPError("url", 405, "Not Allowed", {}, None)
        # OEM action also fails
        oem_action_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        # OEM PATCH also fails
        oem_patch_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        mock_urlopen.side_effect = [
            thermal_get, thermal_patch_405,  # thermal PATCH attempt
            oem_action_404,  # OEM action attempt
            oem_patch_404,  # OEM PATCH attempt
        ]
        # IPMI LAN succeeds
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = ctrl.set_manual_fan_percent(60)
        self.assertTrue(result)
        self.assertEqual(ctrl._working_method, XCCFanController.METHOD_IPMI_LAN)
        self.assertTrue(ctrl.manual_active)


# ===========================================================================
# Restore Tests
# ===========================================================================

class TestXCCFanControllerRestore(unittest.TestCase):
    """restore_auto_control() behaviour."""

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_restore_via_oem_action(self, mock_urlopen):
        """OEM action restore: POST Automatic → success."""
        ctrl = _make_controller(XCCFanController.METHOD_REDFISH_OEM_ACTION)
        ctrl._manual_active = True
        mock_urlopen.side_effect = [_make_urlopen_response(200, {})]

        result = ctrl.restore_auto_control()
        self.assertTrue(result)
        self.assertFalse(ctrl.manual_active)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_restore_via_ipmi_lan_lenovo(self, mock_run):
        """IPMI LAN Lenovo auto restore."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        ctrl._manual_active = True
        ctrl._ipmi_lan_variant = "lenovo_0x3a"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = ctrl.restore_auto_control()
        self.assertTrue(result)
        self.assertFalse(ctrl.manual_active)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    def test_restore_via_ipmi_lan_dell(self, mock_run):
        """IPMI LAN Dell auto restore."""
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        ctrl._manual_active = True
        ctrl._ipmi_lan_variant = "dell_0x30"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = ctrl.restore_auto_control()
        self.assertTrue(result)
        self.assertFalse(ctrl.manual_active)

    def test_restore_no_override_active(self):
        """No manual override was active → returns True immediately."""
        ctrl = _make_controller()
        ctrl._manual_active = False
        result = ctrl.restore_auto_control()
        self.assertTrue(result)

    @patch("core.hardware.xcc_fan_controller.subprocess.run")
    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_restore_all_fail(self, mock_urlopen, mock_run):
        """All restore paths fail → returns False, manual still active."""
        import urllib.error
        ctrl = _make_controller(XCCFanController.METHOD_IPMI_LAN, ipmi_lan=True)
        ctrl._manual_active = True

        err = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
        mock_urlopen.side_effect = err  # All Redfish fails
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        result = ctrl.restore_auto_control()
        self.assertFalse(result)
        self.assertTrue(ctrl.manual_active)


# ===========================================================================
# Read Fan Speeds Tests
# ===========================================================================

class TestXCCFanControllerReadFans(unittest.TestCase):
    """read_fan_speeds() returns parsed fan data."""

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_read_fans_success(self, mock_urlopen):
        """Reads fan data and returns list of dicts."""
        ctrl = _make_controller()
        thermal = _make_urlopen_response(200, _build_thermal_body(4, 56))
        mock_urlopen.side_effect = [thermal]

        fans = ctrl.read_fan_speeds()
        self.assertEqual(len(fans), 4)
        self.assertEqual(fans[0]["name"], "Fan 1")
        self.assertEqual(fans[0]["reading_pct"], 56)
        self.assertEqual(fans[0]["status"], "OK")

    def test_read_fans_not_available(self):
        """Not available → returns empty list."""
        ctrl = _make_controller()
        ctrl._available = False
        self.assertEqual(ctrl.read_fan_speeds(), [])

    @patch("core.hardware.xcc_fan_controller.urllib.request.urlopen")
    def test_connection_timeout_graceful(self, mock_urlopen):
        """Connection timeout → returns empty list, no exception."""
        ctrl = _make_controller()
        mock_urlopen.side_effect = Exception("Connection timed out")
        fans = ctrl.read_fan_speeds()
        self.assertEqual(fans, [])


# ===========================================================================
# Agent Integration Tests
# ===========================================================================

class TestAgentXCCIntegration(unittest.TestCase):
    """Test that agent's set_fan_duty correctly dispatches to XCC."""

    def test_xcc_method_in_set_fan_duty(self):
        """When XCC controller is available and target > 3500 RPM, method = xcc_redfish."""
        import scripts.cooledai_agent as agent

        mock_xcc = MagicMock()
        mock_xcc.available = True
        mock_xcc.set_manual_fan_percent.return_value = True

        old_xcc = agent._XCC_FAN_CONTROLLER
        agent._XCC_FAN_CONTROLLER = mock_xcc
        agent._xcc_override_active = False
        try:
            caps = agent.SensorCapabilities()
            method = agent.set_fan_duty(caps, 60, dry_run=False, hw_rpm=0.0)
            self.assertEqual(method, "xcc_redfish")
            mock_xcc.set_manual_fan_percent.assert_called_once_with(60)
        finally:
            agent._XCC_FAN_CONTROLLER = old_xcc

    def test_xcc_not_triggered_below_threshold(self):
        """When target < 3500 RPM, XCC is not triggered — falls through to 'none'."""
        import scripts.cooledai_agent as agent

        mock_xcc = MagicMock()
        mock_xcc.available = True

        old_xcc = agent._XCC_FAN_CONTROLLER
        agent._XCC_FAN_CONTROLLER = mock_xcc
        agent._xcc_override_active = False
        try:
            caps = agent.SensorCapabilities()
            method = agent.set_fan_duty(caps, 25, dry_run=False, hw_rpm=0.0)
            self.assertEqual(method, "none")
            mock_xcc.set_manual_fan_percent.assert_not_called()
        finally:
            agent._XCC_FAN_CONTROLLER = old_xcc

    def test_xcc_restore_on_low_target(self):
        """When XCC override is active and target drops below 2000 RPM, restore auto."""
        import scripts.cooledai_agent as agent

        mock_xcc = MagicMock()
        mock_xcc.available = True
        mock_xcc.restore_auto_control.return_value = True

        old_xcc = agent._XCC_FAN_CONTROLLER
        old_active = agent._xcc_override_active
        agent._XCC_FAN_CONTROLLER = mock_xcc
        agent._xcc_override_active = True
        try:
            caps = agent.SensorCapabilities()
            method = agent.set_fan_duty(caps, 20, dry_run=False, hw_rpm=0.0)
            mock_xcc.restore_auto_control.assert_called_once()
            self.assertFalse(agent._xcc_override_active)
        finally:
            agent._XCC_FAN_CONTROLLER = old_xcc
            agent._xcc_override_active = old_active


if __name__ == "__main__":
    unittest.main()
