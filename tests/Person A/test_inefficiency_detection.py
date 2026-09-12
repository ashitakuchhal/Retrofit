"""
Unit Tests for HVAC Inefficiency Detection Layer
================================================
Tests hand-crafted known-fault cases from bldg59, normal cases,
TestBed zoning cases, interface contract conformity, and metrics.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "Person A"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from inefficiency_detection import (
    detect_economizer_fault,
    detect_sensor_mismatch,
    detect_ventilation_imbalance,
    detect_poor_zoning,
    detect_inefficiencies,
)

class TestInefficiencyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bldg59_path = Path("data/processed/bldg59_master_hourly_clean.csv")
        cls.testbed_path = Path("data/processed/TestBedClean.csv")

        if cls.bldg59_path.exists():
            cls.bldg59 = pd.read_csv(cls.bldg59_path)
        else:
            cls.bldg59 = None

        if cls.testbed_path.exists():
            cls.testbed = pd.read_csv(cls.testbed_path)
        else:
            cls.testbed = None

    def test_shared_interface_contract(self):
        """Verify return type, key names, and 0-5 integer range."""
        dummy_row = {
            "rtu_oa_damper_avg": 40.0,
            "outdoor_temp_f": 65.0,
            "indoor_temp_f": 72.0,
            "rtu_ma_temp_avg": 69.2,
            "hvac_kw": 35.0,
            "total_occ": 10.0,
            "is_business_hours": 1,
            "T_Room_102": 21.5,
            "T_Room_103": 22.0,
        }
        res = detect_inefficiencies(dummy_row)
        self.assertIsInstance(res, dict)
        required_keys = ["poor_zoning", "ventilation_imbalance",
                         "economizer_fault", "sensor_mismatch"]
        for k in required_keys:
            self.assertIn(k, res)
            self.assertIsInstance(res[k], int)
            self.assertTrue(
                0 <= res[k] <= 5, f"Score for {k} ({res[k]}) is outside 0-5")

    def test_handpicked_stuck_open_row_11475(self):
        """
        Known Fault 1: bldg59 row 11475 (2019-04-24 03:00:00)
        Damper = 97.72%, Outdoor Temp = 75.9°F, HVAC = 70.9 kW.
        Ground Truth: fault_econ_stuck_open == True.
        """
        if self.bldg59 is None:
            self.skipTest("bldg59 dataset not found")

        row = self.bldg59.iloc[11475]
        self.assertTrue(bool(row["fault_econ_stuck_open"]))

        score, details = detect_economizer_fault(row, return_details=True)
        self.assertTrue(details["stuck_open"], "Detector must flag stuck open")
        self.assertGreaterEqual(
            score, 4, "Severity score for stuck open warm air must be >= 4")

        full_scores = detect_inefficiencies(row)
        self.assertGreaterEqual(full_scores["economizer_fault"], 4)

    def test_handpicked_stuck_open_row_12628(self):
        """
        Known Fault 2: bldg59 row 12628 (2019-06-11 04:00:00)
        Damper = 97.75%, Outdoor Temp = 76.5°F, HVAC = 83.5 kW.
        Ground Truth: fault_econ_stuck_open == True.
        """
        if self.bldg59 is None:
            self.skipTest("bldg59 dataset not found")

        row = self.bldg59.iloc[12628]
        self.assertTrue(bool(row["fault_econ_stuck_open"]))

        score, details = detect_economizer_fault(row, return_details=True)
        self.assertTrue(details["stuck_open"])
        self.assertGreaterEqual(score, 4)

    def test_handpicked_stuck_closed_row_7496(self):
        """
        Known Fault 3: bldg59 row 7496 (2018-11-09 08:00:00)
        Damper = 10.0%, Outdoor Temp = 57.7°F (prime economizer), HVAC = 52.4 kW.
        Ground Truth: fault_econ_stuck_closed == True.
        """
        if self.bldg59 is None:
            self.skipTest("bldg59 dataset not found")

        row = self.bldg59.iloc[7496]
        self.assertTrue(bool(row["fault_econ_stuck_closed"]))

        score, details = detect_economizer_fault(row, return_details=True)
        self.assertTrue(details["stuck_closed"],
                        "Detector must flag stuck closed")
        self.assertGreaterEqual(
            score, 3, "Severity score for stuck closed must be >= 3")

    def test_handpicked_sensor_mismatch_row_5550(self):
        """
        Known Fault 4: bldg59 row 5550 (2018-08-20 06:00:00)
        Damper = 0.0%, MA Temp = 52.5°F, Expected = 74.4°F, Deviation = -21.9°F.
        Ground Truth: fault_ma_sensor_mismatch == True.
        """
        if self.bldg59 is None:
            self.skipTest("bldg59 dataset not found")

        row = self.bldg59.iloc[5550]
        self.assertTrue(bool(row["fault_ma_sensor_mismatch"]))

        score, details = detect_sensor_mismatch(row, return_details=True)
        self.assertGreaterEqual(details["abs_deviation_f"], 15.0)
        self.assertGreaterEqual(
            score, 4, "Severity score for |dev| > 15°F must be >= 4")

        full_scores = detect_inefficiencies(row)
        self.assertGreaterEqual(full_scores["sensor_mismatch"], 4)

    def test_handpicked_normal_clean_row_0(self):
        """
        Known Normal Row: bldg59 row 0 (2018-01-01 00:00:00)
        Ground Truth: any_economizer_fault == False.
        """
        if self.bldg59 is None:
            self.skipTest("bldg59 dataset not found")

        row = self.bldg59.iloc[0]
        self.assertFalse(bool(row["any_economizer_fault"]))

        econ_score = detect_economizer_fault(row)
        sensor_score = detect_sensor_mismatch(row)
        self.assertEqual(
            econ_score, 0, "Normal row should have economizer_fault == 0")
        self.assertLessEqual(
            sensor_score, 1, "Normal row should have sensor_mismatch <= 1")

    def test_testbed_poor_zoning(self):
        """TestBed has known high inter-zone temperature spread (mean 3.35°C, max 14.9°C)."""
        if self.testbed is None:
            self.skipTest("testbed dataset not found")

        score, details = detect_poor_zoning(self.testbed, return_details=True)
        self.assertGreaterEqual(
            score, 3, "TestBedClean should register moderate-to-severe poor zoning (>= 3)")
        self.assertGreaterEqual(details["median_zone_spread_c"], 3.0)
        self.assertGreater(details["vav_energy_cv"], 0.5)

    def test_string_identifier_resolution(self):
        """Test calling detect_inefficiencies with string dataset identifiers."""
        res_bldg = detect_inefficiencies("bldg59")
        self.assertIn("economizer_fault", res_bldg)
        res_tb = detect_inefficiencies("testbed")
        self.assertIn("poor_zoning", res_tb)

    def test_return_details_structure(self):
        """Verify the nested structure returned when return_details=True."""
        dummy_row = {
            "rtu_oa_damper_avg": 90.0,
            "outdoor_temp_f": 78.0,
            "indoor_temp_f": 74.0,
            "rtu_ma_temp_avg": 55.0,
            "is_business_hours": 1
        }
        res = detect_inefficiencies(dummy_row, return_details=True)
        self.assertIn("scores", res)
        self.assertIn("flags", res)
        self.assertIn("details", res)
        self.assertIsInstance(res["flags"]["economizer_fault"], bool)
        self.assertIsInstance(res["details"]["economizer"], dict)


if __name__ == "__main__":
    unittest.main()
