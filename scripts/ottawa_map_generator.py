#!/usr/bin/env python3
"""
Ottawa Map Generator for F2MD Framework
Project: 5G Vehicle Misbehavior Detection - SYSC5804/ITEC5910

This script generates SUMO network files for Ottawa City from OpenStreetMap data.
Steps: Download OSM → Convert to SUMO network → Generate traffic → Create config

Usage:
    python3 ottawa_map_generator.py --area downtown --vehicles 200 --duration 300
    python3 ottawa_map_generator.py --area highway417 --vehicles 500 --duration 600
    python3 ottawa_map_generator.py --area full --vehicles 1000 --duration 1800
"""

import os
import sys
import subprocess
import argparse
import urllib.request
from pathlib import Path

# ─── Ottawa Bounding Boxes ───────────────────────────────────────────────────
# Format: (south_lat, west_lon, north_lat, east_lon)
OTTAWA_AREAS = {
    'downtown': {
        'bbox': (45.4150, -75.7050, 45.4350, -75.6850),
        'description': 'Downtown core - Parliament Hill, Rideau Centre, ByWard Market',
        'trip_period': 2,
        'fringe_factor': 5,
    },
    'highway417': {
        'bbox': (45.3800, -75.7500, 45.4200, -75.6500),
        'description': 'Highway 417 corridor - Queensway from Kanata to Downtown',
        'trip_period': 1,
        'fringe_factor': 10,
    },
    'residential': {
        'bbox': (45.3500, -75.7200, 45.3800, -75.6800),
        'description': 'Residential neighborhoods - South Ottawa',
        'trip_period': 5,
        'fringe_factor': 2,
    },
    'full': {
        'bbox': (45.3000, -75.8500, 45.5000, -75.5500),
        'description': 'Greater Ottawa region - Downtown, Kanata, Orleans, Barrhaven',
        'trip_period': 3,
        'fringe_factor': 5,
    },
}


def check_sumo_installed():
    """Verify SUMO is installed and SUMO_HOME is set."""
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("[ERROR] SUMO_HOME environment variable is not set!")
        print("  Run: export SUMO_HOME=\"/usr/share/sumo\"")
        print("  Or add it to ~/.bashrc")
        sys.exit(1)

    # Check for required SUMO tools
    for tool in ['netconvert', 'polyconvert']:
        if not _command_exists(tool):
            print(f"[ERROR] '{tool}' not found. Is SUMO installed correctly?")
            sys.exit(1)

    # Check for randomTrips.py
    random_trips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    if not os.path.exists(random_trips):
        print(f"[ERROR] randomTrips.py not found at {random_trips}")
        sys.exit(1)

    print(f"[OK] SUMO_HOME = {sumo_home}")
    return sumo_home


def _command_exists(cmd):
    """Check if a command exists on the system."""
    try:
        subprocess.run([cmd, '--version'], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_osm_data(area_name, area_config, output_dir):
    """
    Download OpenStreetMap data for the selected Ottawa area.
    Uses the Overpass API.
    """
    osm_file = output_dir / 'ottawa.osm'

    if osm_file.exists():
        print(f"[INFO] OSM file already exists: {osm_file}")
        resp = input("  Re-download? (y/n): ").strip().lower()
        if resp != 'y':
            print("[INFO] Using existing OSM file")
            return osm_file

    south, west, north, east = area_config['bbox']
    print(f"\n[STEP 1] Downloading OSM data for Ottawa ({area_name})...")
    print(f"  Area: {area_config['description']}")
    print(f"  Bounding box: S={south}, W={west}, N={north}, E={east}")

    # Overpass API URL - bbox format is: west,south,east,north
    url = f"https://overpass-api.de/api/map?bbox={west},{south},{east},{north}"

    try:
        print(f"  Downloading from Overpass API...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, str(osm_file))
        file_size = osm_file.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] OSM data saved: {osm_file} ({file_size:.1f} MB)")
        return osm_file
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\n  Alternative: Download manually from https://www.openstreetmap.org/export")
        print(f"  Set the bounding box to: {west},{south},{east},{north}")
        print(f"  Save the file as: {osm_file}")
        sys.exit(1)


def convert_to_network(osm_file, output_dir):
    """
    Convert OSM data to SUMO network format using netconvert.
    """
    net_file = output_dir / 'ottawa.net.xml'
    print(f"\n[STEP 2] Converting OSM to SUMO network...")

    cmd = [
        'netconvert',
        '--osm-files', str(osm_file),
        '--output-file', str(net_file),
        # Geometry and topology optimization
        '--geometry.remove',
        '--ramps.guess',
        '--junctions.join',
        # Traffic light handling
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        # Output options
        '--output.street-names',
        '--output.original-names',
        # Junction detail
        '--junctions.corner-detail', '5',
        # Network cleanup
        '--roundabouts.guess',
        '--remove-edges.isolated',
        # Keep only road edges (no railways, waterways, etc.)
        '--keep-edges.by-vclass', 'passenger',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARNING] netconvert had warnings:\n{result.stderr[:500]}")

        if net_file.exists():
            file_size = net_file.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] SUMO network saved: {net_file} ({file_size:.1f} MB)")

            # Count network statistics
            _print_network_stats(net_file)
            return net_file
        else:
            print("[ERROR] Network file was not created")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] netconvert timed out (>120s). Try a smaller area.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] netconvert failed: {e}")
        sys.exit(1)


def _print_network_stats(net_file):
    """Print basic statistics about the generated network."""
    try:
        with open(net_file, 'r') as f:
            content = f.read()
        edges = content.count('<edge id=')
        junctions = content.count('<junction id=')
        print(f"  Network statistics:")
        print(f"    Roads (edges): {edges}")
        print(f"    Intersections (junctions): {junctions}")
    except Exception:
        pass


def generate_polygons(osm_file, net_file, output_dir, sumo_home):
    """
    Generate building polygons for visualization in SUMO-GUI.
    This is optional - the simulation works without it.
    """
    poly_file = output_dir / 'ottawa.poly.xml'
    print(f"\n[STEP 3] Generating building polygons (optional)...")

    typemap = os.path.join(sumo_home, 'data', 'typemap', 'osmPolyconvert.typ.xml')
    if not os.path.exists(typemap):
        print(f"[WARNING] Typemap not found at {typemap}, skipping polygons")
        return None

    cmd = [
        'polyconvert',
        '--osm-files', str(osm_file),
        '--net-file', str(net_file),
        '--output-file', str(poly_file),
        '--type-file', typemap,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if poly_file.exists():
            print(f"[SUCCESS] Polygons saved: {poly_file}")
            return poly_file
        else:
            print("[WARNING] Polygon file not created (non-critical)")
            return None
    except Exception as e:
        print(f"[WARNING] polyconvert failed (non-critical): {e}")
        return None


def generate_traffic(net_file, output_dir, area_config, num_vehicles, duration, sumo_home):
    """
    Generate realistic traffic demand using SUMO's randomTrips.py.
    """
    route_file = output_dir / 'ottawa.rou.xml'
    trips_file = output_dir / 'ottawa.trips.xml'
    print(f"\n[STEP 4] Generating traffic demand...")
    print(f"  Vehicles: {num_vehicles}")
    print(f"  Duration: {duration} seconds")

    # Calculate trip period from desired vehicle count and duration
    # period = duration / num_vehicles gives roughly the right number
    period = max(0.5, duration / num_vehicles)

    random_trips_path = os.path.join(sumo_home, 'tools', 'randomTrips.py')

    cmd = [
        'python3', random_trips_path,
        '--net-file', str(net_file),
        '--output-trip-file', str(trips_file),
        '--route-file', str(route_file),
        '--end', str(duration),
        '--period', str(period),
        '--fringe-factor', str(area_config['fringe_factor']),
        '--vehicle-class', 'passenger',
        '--validate',
        '--seed', '42',  # Reproducible results
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[WARNING] randomTrips.py warnings:\n{result.stderr[:500]}")

        if route_file.exists():
            # Count generated vehicles
            with open(route_file, 'r') as f:
                content = f.read()
            vehicle_count = content.count('<vehicle id=')
            print(f"[SUCCESS] Traffic demand saved: {route_file}")
            print(f"  Generated vehicles: {vehicle_count}")
            return route_file
        else:
            print("[ERROR] Route file was not created")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] Traffic generation timed out. Try fewer vehicles.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Traffic generation failed: {e}")
        sys.exit(1)


def create_sumo_config(output_dir, net_file, route_file, poly_file, duration):
    """
    Create the SUMO configuration file (.sumocfg).
    """
    config_file = output_dir / 'ottawa.sumocfg'
    print(f"\n[STEP 5] Creating SUMO configuration file...")

    poly_line = ""
    if poly_file and poly_file.exists():
        poly_line = f'        <additional-files value="ottawa.poly.xml"/>'

    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="ottawa.net.xml"/>
        <route-files value="ottawa.rou.xml"/>
{poly_line}
    </input>

    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="0.1"/>
    </time>

    <processing>
        <collision.action value="warn"/>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="false"/>
    </report>

</configuration>
"""

    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"[SUCCESS] SUMO config saved: {config_file}")
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate Ottawa SUMO Network for F2MD Framework'
    )
    parser.add_argument(
        '--area',
        choices=['downtown', 'highway417', 'residential', 'full'],
        default='downtown',
        help='Ottawa area to simulate (default: downtown)'
    )
    parser.add_argument(
        '--vehicles',
        type=int,
        default=200,
        help='Number of vehicles (default: 200)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Simulation duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--output-dir',
        default='ottawa_sumo',
        help='Output directory (default: ottawa_sumo)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip OSM download (use existing ottawa.osm)'
    )

    args = parser.parse_args()
    area_config = OTTAWA_AREAS[args.area]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    print("=" * 60)
    print("  Ottawa Map Generator - F2MD Framework")
    print("  SYSC5804/ITEC5910 - 5G Vehicle Misbehavior Detection")
    print("=" * 60)
    print(f"  Area:      {args.area.upper()} - {area_config['description']}")
    print(f"  Vehicles:  {args.vehicles}")
    print(f"  Duration:  {args.duration}s ({args.duration//60} min {args.duration%60}s)")
    print(f"  Output:    {output_dir}/")
    print("=" * 60)

    # Check SUMO installation
    sumo_home = check_sumo_installed()

    # Step 1: Download OSM data
    if args.skip_download:
        osm_file = output_dir / 'ottawa.osm'
        if not osm_file.exists():
            print("[ERROR] --skip-download used but ottawa.osm not found!")
            sys.exit(1)
        print(f"[INFO] Using existing OSM file: {osm_file}")
    else:
        osm_file = download_osm_data(args.area, area_config, output_dir)

    # Step 2: Convert to SUMO network
    net_file = convert_to_network(osm_file, output_dir)

    # Step 3: Generate polygons (optional, for visualization)
    poly_file = generate_polygons(osm_file, net_file, output_dir, sumo_home)

    # Step 4: Generate traffic demand
    route_file = generate_traffic(
        net_file, output_dir, area_config,
        args.vehicles, args.duration, sumo_home
    )

    # Step 5: Create SUMO config
    config_file = create_sumo_config(
        output_dir, net_file, route_file, poly_file, args.duration
    )

    # Final summary
    print("\n" + "=" * 60)
    print("  Ottawa SUMO Scenario Generated Successfully!")
    print("=" * 60)
    print(f"\n  Files created in {output_dir}/:")
    print(f"    ottawa.osm       - Raw OpenStreetMap data")
    print(f"    ottawa.net.xml   - SUMO road network")
    print(f"    ottawa.rou.xml   - Vehicle routes and traffic")
    if poly_file:
        print(f"    ottawa.poly.xml  - Building polygons")
    print(f"    ottawa.sumocfg   - SUMO configuration")
    print(f"\n  Next steps:")
    print(f"    1. Verify in SUMO GUI:")
    print(f"       sumo-gui -c {config_file}")
    print(f"    2. Run headless test:")
    print(f"       sumo -c {config_file} --duration-log.statistics --no-step-log")
    print(f"    3. Move to Phase 3: Integrate with F2MD")
    print("=" * 60)


if __name__ == '__main__':
    main()
