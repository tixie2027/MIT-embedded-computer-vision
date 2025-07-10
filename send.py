#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
#    this is an UART-LoRa device and thers is an firmware on Module
#    users can transfer or receive the data directly by UART and dont
#    need to set parameters like coderate,spread factor,etc.
#    |============================================ |
#    |   It does not suport LoRaWAN protocol !!!   |
#    | ============================================|
#   
#    This script is mainly for Raspberry Pi 3B+, 4B, and Zero series
#    Since PC/Laptop does not have GPIO to control HAT, it should be configured by
#    GUI and while setting the jumpers, 
#    Please refer to another script pc_main.py
#

import sys
import sx126x
import threading
import time
import select
import termios
import tty
from threading import Timer

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())


node = sx126x.sx126x(serial_num = "/dev/serial0",freq=868,addr=0,power=22,rssi=True,air_speed=2400,relay=False)

CHUNK_SIZE = 64  # 8 hex chars = 4 bytes
HEADER_BYTES = [255, 255, 18]  # Preamble + packet type

def load_hex_strings(filename="embeddings.txt"):
    with open(filename, "r") as f:
        raw = f.read().strip()
    return [s.strip() for s in raw.split(",")]

def chunk_hex_string(hex_str, chunk_size):
    return [hex_str[i:i + chunk_size] for i in range(0, len(hex_str), chunk_size)]


def main():
    try:
        with open("embeddings.txt", "r") as f:
            for line in f:
                if line.strip() == "":
                    continue  # skip empty lines
                data = bytes([255]) + bytes([255]) + bytes([18]) + bytes([node.addr >> 8]) + bytes([node.addr & 0xff]) + bytes([node.offset_freq]) + line.strip().encode()
                node.send(data)
                print(f"Sent: {line.strip()}")
                time.sleep(0.1)  # slight delay between sends

    except:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
if __name__ == "__main__":
    main()