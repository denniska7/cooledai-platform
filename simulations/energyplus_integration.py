"""
EnergyPlus Integration for CoolingAI Simulator

This module provides integration with EnergyPlus building energy simulation:
- IDF (Input Data File) generation for data centers
- Weather file manipulation
- EnergyPlus simulation execution
- PUE calculation from simulation results
- Validation against PINN predictions

Requirements:
    pip install eppy
    EnergyPlus must be installed on the system
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
import pandas as pd
import numpy as np

try:
    from eppy import modeleditor
    from eppy.modeleditor import IDF
    EPPY_AVAILABLE = True
except ImportError:
    EPPY_AVAILABLE = False
    print("WARNING: eppy not installed. Install with: pip install eppy")


class DataCenterIDFGenerator:
    """
    Generate EnergyPlus IDF files for data center facilities.

    Creates a complete building model with:
    - Data center zones
    - HVAC systems (CRAC units, chillers)
    - IT equipment loads
    - Outdoor air economization
    """

    def __init__(self, energyplus_version: str = '23.2.0'):
        """
        Initialize IDF generator.

        Args:
            energyplus_version: EnergyPlus version (e.g., '23.2.0')
        """
        if not EPPY_AVAILABLE:
            raise ImportError("eppy library not available")

        self.energyplus_version = energyplus_version
        self.idd_path = self._find_idd_file()

        if self.idd_path:
            IDF.setiddname(self.idd_path)

    def _find_idd_file(self) -> Optional[str]:
        """
        Locate Energy+.idd file on the system.

        Returns:
            Path to IDD file or None if not found
        """
        # Common locations for EnergyPlus installation
        possible_paths = [
            f"/usr/local/EnergyPlus-{self.energyplus_version}/Energy+.idd",
            f"/Applications/EnergyPlus-{self.energyplus_version}/Energy+.idd",
            f"C:/EnergyPlusV{self.energyplus_version.replace('.', '-')}/Energy+.idd",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        print(f"Warning: Could not find Energy+.idd for version {self.energyplus_version}")
        return None

    def create_base_idf(
        self,
        length: float = 20.0,
        width: float = 16.0,
        height: float = 3.0,
        name: str = "DataCenter_Base"
    ) -> IDF:
        """
        Create base IDF with data center geometry.

        Args:
            length: Building length [m]
            width: Building width [m]
            height: Building height [m]
            name: Building name

        Returns:
            IDF object
        """
        idf = IDF()

        # Building
        idf.newidfobject(
            'BUILDING',
            Name=name,
            North_Axis=0,
            Terrain='City',
            Loads_Convergence_Tolerance_Value=0.04,
            Temperature_Convergence_Tolerance_Value=0.4,
            Solar_Distribution='FullInteriorAndExterior',
            Maximum_Number_of_Warmup_Days=25,
            Minimum_Number_of_Warmup_Days=6
        )

        # Timestep (4 timesteps per hour = 15 min intervals)
        idf.newidfobject('TIMESTEP', Number_of_Timesteps_per_Hour=4)

        # Run period (annual simulation)
        idf.newidfobject(
            'RUNPERIOD',
            Name='Annual',
            Begin_Month=1,
            Begin_Day_of_Month=1,
            End_Month=12,
            End_Day_of_Month=31,
            Day_of_Week_for_Start_Day='Sunday',
            Use_Weather_File_Holidays_and_Special_Days='Yes',
            Use_Weather_File_Daylight_Saving_Period='Yes',
            Apply_Weekend_Holiday_Rule='No',
            Use_Weather_File_Rain_Indicators='Yes',
            Use_Weather_File_Snow_Indicators='Yes'
        )

        # Create zone (data center space)
        zone = idf.newidfobject('ZONE', Name='DataCenter_Zone')

        # Create simple box geometry
        self._add_box_geometry(idf, 'DataCenter_Zone', length, width, height)

        return idf

    def _add_box_geometry(
        self,
        idf: IDF,
        zone_name: str,
        length: float,
        width: float,
        height: float
    ):
        """Add simple box geometry for the data center."""

        # Floor
        idf.newidfobject(
            'BUILDINGSURFACE:DETAILED',
            Name='Floor',
            Surface_Type='Floor',
            Construction_Name='Floor_Construction',
            Zone_Name=zone_name,
            Outside_Boundary_Condition='Ground',
            Number_of_Vertices=4,
            Vertex_1_Xcoordinate=0, Vertex_1_Ycoordinate=0, Vertex_1_Zcoordinate=0,
            Vertex_2_Xcoordinate=length, Vertex_2_Ycoordinate=0, Vertex_2_Zcoordinate=0,
            Vertex_3_Xcoordinate=length, Vertex_3_Ycoordinate=width, Vertex_3_Zcoordinate=0,
            Vertex_4_Xcoordinate=0, Vertex_4_Ycoordinate=width, Vertex_4_Zcoordinate=0
        )

        # Ceiling
        idf.newidfobject(
            'BUILDINGSURFACE:DETAILED',
            Name='Ceiling',
            Surface_Type='Roof',
            Construction_Name='Roof_Construction',
            Zone_Name=zone_name,
            Outside_Boundary_Condition='Outdoors',
            Number_of_Vertices=4,
            Vertex_1_Xcoordinate=0, Vertex_1_Ycoordinate=0, Vertex_1_Zcoordinate=height,
            Vertex_2_Xcoordinate=0, Vertex_2_Ycoordinate=width, Vertex_2_Zcoordinate=height,
            Vertex_3_Xcoordinate=length, Vertex_3_Ycoordinate=width, Vertex_3_Zcoordinate=height,
            Vertex_4_Xcoordinate=length, Vertex_4_Ycoordinate=0, Vertex_4_Zcoordinate=height
        )

        # Walls (4 sides)
        walls = [
            ('Wall_South', 0, 0, 0, length, 0, 0, length, 0, height, 0, 0, height),
            ('Wall_East', length, 0, 0, length, width, 0, length, width, height, length, 0, height),
            ('Wall_North', length, width, 0, 0, width, 0, 0, width, height, length, width, height),
            ('Wall_West', 0, width, 0, 0, 0, 0, 0, 0, height, 0, width, height)
        ]

        for name, *coords in walls:
            idf.newidfobject(
                'BUILDINGSURFACE:DETAILED',
                Name=name,
                Surface_Type='Wall',
                Construction_Name='Wall_Construction',
                Zone_Name=zone_name,
                Outside_Boundary_Condition='Outdoors',
                Number_of_Vertices=4,
                Vertex_1_Xcoordinate=coords[0], Vertex_1_Ycoordinate=coords[1], Vertex_1_Zcoordinate=coords[2],
                Vertex_2_Xcoordinate=coords[3], Vertex_2_Ycoordinate=coords[4], Vertex_2_Zcoordinate=coords[5],
                Vertex_3_Xcoordinate=coords[6], Vertex_3_Ycoordinate=coords[7], Vertex_3_Zcoordinate=coords[8],
                Vertex_4_Xcoordinate=coords[9], Vertex_4_Ycoordinate=coords[10], Vertex_4_Zcoordinate=coords[11]
            )

    def add_it_equipment(
        self,
        idf: IDF,
        zone_name: str,
        power_density: float = 5000.0  # W/m²
    ):
        """
        Add IT equipment loads to the zone.

        Args:
            idf: IDF object
            zone_name: Name of the zone
            power_density: Power density [W/m²]
        """
        # IT equipment schedule (24/7 operation)
        idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='IT_Equipment_Schedule',
            Schedule_Type_Limits_Name='',
            Hourly_Value=1.0
        )

        # Electric equipment for IT loads
        idf.newidfobject(
            'ELECTRICEQUIPMENT',
            Name='IT_Equipment',
            Zone_or_ZoneList_Name=zone_name,
            Schedule_Name='IT_Equipment_Schedule',
            Design_Level_Calculation_Method='Watts/Area',
            Watts_per_Zone_Floor_Area=power_density,
            Fraction_Latent=0.0,  # IT equipment is mostly sensible heat
            Fraction_Radiant=0.3,
            Fraction_Lost=0.0
        )

    def add_crac_system(
        self,
        idf: IDF,
        zone_name: str,
        supply_temp: float = 18.0  # °C
    ):
        """
        Add CRAC (Computer Room Air Conditioner) system.

        Args:
            idf: IDF object
            zone_name: Name of the zone
            supply_temp: Supply air temperature [°C]
        """
        # Thermostat
        idf.newidfobject(
            'HVACTEMPLATE:THERMOSTAT',
            Name='DataCenter_Thermostat',
            Heating_Setpoint_Schedule_Name='',
            Constant_Heating_Setpoint=15.0,  # Heating rarely needed
            Cooling_Setpoint_Schedule_Name='',
            Constant_Cooling_Setpoint=24.0  # Target zone temperature
        )

        # CRAC system (using HVACTemplate for simplicity)
        idf.newidfobject(
            'HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM',
            Zone_Name=zone_name,
            Template_Thermostat_Name='DataCenter_Thermostat',
            System_Availability_Schedule_Name='',
            Maximum_Heating_Supply_Air_Temperature=50.0,
            Minimum_Cooling_Supply_Air_Temperature=supply_temp,
            Maximum_Heating_Supply_Air_Humidity_Ratio=0.0156,
            Minimum_Cooling_Supply_Air_Humidity_Ratio=0.0077,
            Heating_Limit='NoLimit',
            Cooling_Limit='NoLimit',
            Dehumidification_Control_Type='None',
            Cooling_Sensible_Heat_Ratio=0.9,  # Mostly sensible cooling
            Dehumidification_Setpoint=50.0,  # % RH
            Humidification_Setpoint=30.0  # % RH
        )

    def set_outdoor_temperature(
        self,
        weather_file_path: str,
        temperature_offset: float = 0.0
    ) -> str:
        """
        Modify weather file to add temperature offset.

        Args:
            weather_file_path: Path to EPW weather file
            temperature_offset: Temperature offset to add [°C]

        Returns:
            Path to modified weather file
        """
        # Read weather file
        with open(weather_file_path, 'r') as f:
            lines = f.readlines()

        # Modify temperature data (lines after header)
        # EPW format: Year,Month,Day,Hour,Minute,Data Source,Temp,Dewpoint,...
        modified_lines = []

        for i, line in enumerate(lines):
            if i < 8:  # Header lines
                modified_lines.append(line)
            else:
                parts = line.split(',')
                if len(parts) > 6:
                    # Modify dry bulb temperature (index 6)
                    try:
                        temp = float(parts[6]) + temperature_offset
                        parts[6] = f"{temp:.1f}"
                        modified_lines.append(','.join(parts))
                    except ValueError:
                        modified_lines.append(line)
                else:
                    modified_lines.append(line)

        # Save modified weather file
        output_path = weather_file_path.replace('.epw', f'_offset_{temperature_offset}C.epw')
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

        return output_path

    def save_idf(self, idf: IDF, output_path: str):
        """Save IDF file."""
        idf.saveas(output_path)
        print(f"IDF saved to: {output_path}")


class EnergyPlusSimulator:
    """
    Run EnergyPlus simulations and extract results.
    """

    def __init__(self, energyplus_exe: Optional[str] = None):
        """
        Initialize simulator.

        Args:
            energyplus_exe: Path to EnergyPlus executable
        """
        self.energyplus_exe = energyplus_exe or self._find_energyplus()

    def _find_energyplus(self) -> str:
        """Find EnergyPlus executable."""
        possible_paths = [
            "/usr/local/EnergyPlus-23-2-0/energyplus",
            "/Applications/EnergyPlus-23-2-0/energyplus",
            "C:/EnergyPlusV23-2-0/energyplus.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return "energyplus"  # Assume it's in PATH

    def run_simulation(
        self,
        idf_path: str,
        weather_file: str,
        output_dir: str
    ) -> bool:
        """
        Run EnergyPlus simulation.

        Args:
            idf_path: Path to IDF file
            weather_file: Path to EPW weather file
            output_dir: Output directory for results

        Returns:
            True if simulation successful
        """
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            self.energyplus_exe,
            '-w', weather_file,
            '-d', output_dir,
            idf_path
        ]

        print(f"Running EnergyPlus simulation...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✓ Simulation completed successfully")
                return True
            else:
                print(f"✗ Simulation failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"✗ Error running simulation: {e}")
            return False

    def extract_pue(self, output_dir: str) -> Optional[float]:
        """
        Calculate PUE from EnergyPlus output.

        Args:
            output_dir: Directory containing simulation outputs

        Returns:
            PUE value or None if not available
        """
        # Read meter outputs
        meter_file = os.path.join(output_dir, 'eplusout.mtr')

        if not os.path.exists(meter_file):
            print(f"Meter file not found: {meter_file}")
            return None

        # Parse meter file to get IT power and cooling power
        # This is simplified - actual implementation would need detailed parsing

        return None  # Placeholder


# Example usage
def create_datacenter_model_example():
    """Example: Create a data center IDF model."""

    print("CoolingAI Simulator - EnergyPlus Integration Example")
    print("=" * 60)

    if not EPPY_AVAILABLE:
        print("ERROR: eppy not installed")
        print("Install with: pip install eppy")
        return

    # Initialize generator
    generator = DataCenterIDFGenerator()

    if not generator.idd_path:
        print("WARNING: EnergyPlus IDD file not found")
        print("Please install EnergyPlus or set IDD path manually")
        return

    # Create base IDF
    print("\n1. Creating base data center model...")
    idf = generator.create_base_idf(
        length=20.0,  # 20m x 16m data center
        width=16.0,
        height=3.0,
        name="CoolingAI_DataCenter"
    )

    # Add IT equipment (high density - 5 kW/m²)
    print("2. Adding IT equipment loads...")
    generator.add_it_equipment(
        idf,
        zone_name='DataCenter_Zone',
        power_density=5000.0  # 5 kW/m²
    )

    # Add CRAC system
    print("3. Adding CRAC cooling system...")
    generator.add_crac_system(
        idf,
        zone_name='DataCenter_Zone',
        supply_temp=18.0  # 18°C supply air
    )

    # Save IDF
    output_path = "simulations/idf_templates/datacenter_base.idf"
    print(f"\n4. Saving IDF file...")
    generator.save_idf(idf, output_path)

    print("\n" + "=" * 60)
    print("✓ Data center model created successfully!")
    print(f"✓ IDF file: {output_path}")
    print("\nNext steps:")
    print("1. Get a weather file (.epw) for your location")
    print("2. Run simulation: energyplus -w weather.epw datacenter_base.idf")
    print("3. Analyze results to calculate PUE")


if __name__ == "__main__":
    create_datacenter_model_example()
