"""
Demo Script for Automation Controls
Runs a scripted scenario demonstrating the system capabilities
"""
import asyncio
import httpx
import time
from datetime import datetime

PLC_API = "http://localhost:8001"
PLANT_SIM_API = "http://localhost:8002"


class DemoRunner:
    """Orchestrates the demo scenario"""
    
    def __init__(self):
        self.client = httpx.Client(timeout=5.0)
    
    def log(self, message: str):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def wait(self, seconds: float, message: str = ""):
        """Wait with optional message"""
        if message:
            self.log(message)
        time.sleep(seconds)
    
    def call_plc(self, method: str, endpoint: str, data: dict = None):
        """Call PLC API"""
        try:
            url = f"{PLC_API}{endpoint}"
            if method == "GET":
                response = self.client.get(url)
            elif method == "POST":
                response = self.client.post(url, json=data)
            else:
                return None
            
            if response.status_code in [200, 201]:
                return response.json()
            else:
                self.log(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            self.log(f"Connection error: {e}")
            return None
    
    def call_plant_sim(self, method: str, endpoint: str, data: dict = None):
        """Call Plant Simulator API"""
        try:
            url = f"{PLANT_SIM_API}{endpoint}"
            if method == "GET":
                response = self.client.get(url)
            elif method == "POST":
                response = self.client.post(url, json=data)
            else:
                return None
            
            if response.status_code in [200, 201]:
                return response.json()
            else:
                return None
        except Exception as e:
            self.log(f"Plant sim error: {e}")
            return None
    
    def get_status(self):
        """Get PLC status"""
        return self.call_plc("GET", "/status")
    
    def get_alarms(self):
        """Get alarms"""
        result = self.call_plc("GET", "/alarms")
        return result.get("alarms", []) if result else []
    
    def run_demo(self):
        """Run the full demo scenario"""
        self.log("=" * 60)
        self.log("AUTOMATION CONTROLS DEMO - SCRIPTED SCENARIO")
        self.log("=" * 60)
        
        # Step 1: Initial state check
        self.log("\n=== STEP 1: System Initialization ===")
        status = self.get_status()
        if status:
            self.log(f"PLC Mode: {status.get('mode')}")
            self.log(f"Running: {status.get('running')}")
            self.log(f"Scan Time: {status.get('scan_time_ms')} ms")
        else:
            self.log("ERROR: Cannot connect to PLC")
            return
        
        self.wait(2)
        
        # Step 2: Switch to AUTO mode
        self.log("\n=== STEP 2: Switch to AUTO Mode ===")
        result = self.call_plc("POST", "/mode", {"mode": "AUTO"})
        if result:
            self.log("✓ Switched to AUTO mode")
        self.wait(2)
        
        # Step 3: Start the system
        self.log("\n=== STEP 3: Start System ===")
        result = self.call_plc("POST", "/command/start")
        if result:
            self.log("✓ START command executed")
            self.log("  Conveyor starting...")
        self.wait(3)
        
        status = self.get_status()
        if status and status.get('running'):
            self.log("✓ System is now RUNNING")
        
        # Step 4: Normal operation
        self.log("\n=== STEP 4: Normal Operation ===")
        self.log("Conveyor running in AUTO mode...")
        self.log("Products being processed...")
        self.wait(5, "Monitoring operation for 5 seconds...")
        
        status = self.get_status()
        if status:
            self.log(f"  Scan count: {status.get('scan_count')}")
            self.log(f"  Scan time: {status.get('scan_time_ms')} ms")
        
        # Step 5: Inject JAM fault
        self.log("\n=== STEP 5: Fault Scenario - Conveyor Jam ===")
        self.log("⚠ Injecting CONVEYOR JAM fault...")
        result = self.call_plant_sim("POST", "/fault", {
            "fault_type": "jam",
            "active": True
        })
        if result:
            self.log("✓ Jam fault injected")
        
        self.wait(6, "Waiting for jam detection (5 second timeout)...")
        
        # Check for alarm
        status = self.get_status()
        alarms = self.get_alarms()
        
        if status and status.get('mode') == 'FAULT':
            self.log("✓ System entered FAULT mode")
        
        if alarms:
            self.log(f"✓ {len(alarms)} alarm(s) raised:")
            for alarm in alarms:
                self.log(f"  - {alarm.get('tag')}: {alarm.get('message')}")
        
        self.wait(3)
        
        # Step 6: Operator acknowledges alarm
        self.log("\n=== STEP 6: Operator Response ===")
        for alarm in alarms:
            if alarm.get('state') == 'ACTIVE':
                tag = alarm.get('tag')
                self.log(f"Acknowledging alarm: {tag}")
                result = self.call_plc("POST", "/alarms/acknowledge", {"tag": tag})
                if result:
                    self.log(f"✓ Alarm {tag} acknowledged")
        
        self.wait(2)
        
        # Step 7: Clear jam
        self.log("\n=== STEP 7: Clear Jam ===")
        self.log("Operator physically clears jammed product...")
        result = self.call_plant_sim("POST", "/fault", {
            "fault_type": "jam",
            "active": False
        })
        if result:
            self.log("✓ Jam cleared in simulator")
        
        self.wait(2)
        
        # Step 8: Reset system
        self.log("\n=== STEP 8: Reset System ===")
        self.log("Executing RESET command...")
        result = self.call_plc("POST", "/command/reset")
        if result:
            self.log("✓ System RESET successful")
            status = self.get_status()
            if status:
                self.log(f"  New mode: {status.get('mode')}")
        else:
            self.log("✗ Reset failed (may need to clear E-Stop first)")
        
        self.wait(2)
        
        # Step 9: Switch back to AUTO
        self.log("\n=== STEP 9: Resume Operation ===")
        self.log("Switching to AUTO mode...")
        result = self.call_plc("POST", "/mode", {"mode": "AUTO"})
        if result:
            self.log("✓ AUTO mode set")
        
        self.wait(1)
        
        self.log("Starting system...")
        result = self.call_plc("POST", "/command/start")
        if result:
            self.log("✓ System restarted successfully")
        
        self.wait(5, "Monitoring resumed operation...")
        
        status = self.get_status()
        if status and status.get('running'):
            self.log("✓ System operating normally")
        
        # Step 10: Stop system
        self.log("\n=== STEP 10: Controlled Shutdown ===")
        self.log("Executing STOP command...")
        result = self.call_plc("POST", "/command/stop")
        if result:
            self.log("✓ System stopped")
        
        self.wait(2)
        
        # Final status
        self.log("\n=== DEMO COMPLETE ===")
        status = self.get_status()
        if status:
            self.log(f"Final Mode: {status.get('mode')}")
            self.log(f"Running: {status.get('running')}")
            self.log(f"Total Scans: {status.get('scan_count')}")
            self.log(f"Average Scan Time: {status.get('scan_time_ms')} ms")
        
        alarms = self.get_alarms()
        active_alarms = [a for a in alarms if a.get('state') == 'ACTIVE']
        self.log(f"Active Alarms: {len(active_alarms)}")
        
        self.log("\n" + "=" * 60)
        self.log("Demo scenario completed successfully!")
        self.log("=" * 60)


def main():
    """Main entry point"""
    print("\nAutomation Controls Demo Script")
    print("================================\n")
    print("This script will demonstrate:")
    print("  1. System startup in AUTO mode")
    print("  2. Normal operation")
    print("  3. Fault detection (conveyor jam)")
    print("  4. Alarm handling")
    print("  5. Fault clearance and recovery")
    print("  6. Controlled shutdown")
    print("\nMake sure the following services are running:")
    print("  - Soft PLC API (port 8001)")
    print("  - Plant Simulator (port 8002)")
    print("\nPress Enter to start demo...")
    input()
    
    demo = DemoRunner()
    demo.run_demo()
    
    print("\nDemo complete! Check the HMI dashboard for:")
    print("  - Event log")
    print("  - Alarm history")
    print("  - Trend data")
    print("\nOpen HMI at: http://localhost:8501")


if __name__ == "__main__":
    main()

