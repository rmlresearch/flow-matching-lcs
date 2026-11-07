"""Servo bus diagnostic (nothing moves): every ID at 1 Mbps, other baud rates, the raw bytes a ping gets
back, and the adapter's modem lines. Silence everywhere = the motors have no power or no data line;
the adapter itself answers to the OS regardless.   python busprobe7.py [/dev/ttyACM0]"""
import glob, os, sys, time
import serial

SIMLAB = os.environ.get("SIMLAB", os.path.expanduser("~/simlab"))
sys.path.insert(0, os.path.join(SIMLAB, "workspace", "skills"))
from so101.feetech import FeetechBus, _packet, INSTR_PING  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else (sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*")))[0]
print("port", port)
b = FeetechBus(port, 1000000, timeout=0.02)
t0 = time.time()
found = b.scan(range(0, 254))
b.close()
print(f"1 Mbps, IDs 0-253 answering: {found} ({time.time() - t0:.1f}s)")
for baud in (500000, 250000, 128000, 115200, 57600, 38400, 19200):
    s = serial.Serial(port, baud, timeout=0.05)
    s.reset_input_buffer()
    s.write(_packet(1, INSTR_PING))
    raw = s.read(32)
    s.close()
    bb = FeetechBus(port, baud, timeout=0.02)
    f = bb.scan(range(1, 13))
    bb.close()
    print(f"baud {baud}: IDs 1-12 answering {f}; raw reply to ping(1): {raw.hex() if raw else '(nothing)'}")
s = serial.Serial(port, 1000000, timeout=0.05)
print("adapter modem lines CTS/DSR/CD/RI:", s.cts, s.dsr, s.cd, s.ri)
s.close()
