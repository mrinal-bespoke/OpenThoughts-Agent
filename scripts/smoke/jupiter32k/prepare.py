"""Prepare only this smoke's temporary secrets and private login-node proxy."""

import base64
import hashlib
import json
import os
import secrets
import socket
import ipaddress
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from lifecycle import validate_lifetime

ROOT = Path(__file__).resolve().parent
BASE = Path("/e/scratch/reformo/kumar37/jupiter-vista-parity-20261018")
os.umask(0o077)


def write(name, data):
    with (ROOT / name).open("x") as f:
        json.dump(data, f, indent=2)


def start():
    write("proxy-start-intent.json", {"epoch": time.time()})
    values = {}
    for line in (BASE / "network/runtime.env").read_text().splitlines():
        line = line.strip().removeprefix("export ")
        key, sep, value = line.partition("=")
        if sep and key.strip() in ("DAYTONA_DATA_API_KEY", "DAYTONA_API_KEY"):
            parts = shlex.split(value, comments=True)
            if len(parts) == 1:
                values[key.strip()] = parts[0]
    key = values.get("DAYTONA_DATA_API_KEY") or values.get("DAYTONA_API_KEY")
    if not key:
        raise RuntimeError("Required existing credential absent")
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        if ipaddress.ip_address(item[4][0]) in ipaddress.ip_network("10.128.0.0/16")
    }
    if len(addresses) != 1:
        raise RuntimeError("Cannot identify unique internal login address")
    write(
        "private.json",
        {
            "bind": addresses.pop(),
            "password": secrets.token_hex(24),
            "daytona_key": key,
        },
    )
    with (ROOT / "proxy.log").open("x") as log:
        proc = subprocess.Popen(
            ["/usr/bin/python3", str(ROOT / "connect_proxy.py"), str(ROOT)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    write(
        "proxy-process.json",
        {
            "pid": proc.pid,
            "script_sha256": hashlib.sha256(
                (ROOT / "connect_proxy.py").read_bytes()
            ).hexdigest(),
        },
    )
    print(json.dumps({"proxy_pid": proc.pid}), flush=True)


def check():
    ready = json.loads((ROOT / "proxy-ready.json").read_text())
    private = json.loads((ROOT / "private.json").read_text())
    validate_lifetime(ready, time.time())
    proxy = f"http://{ready['host']}:{ready['port']}"
    auth = (
        "Basic " + base64.b64encode(("smoke:" + private["password"]).encode()).decode()
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy}))
    req = urllib.request.Request(
        "https://app.daytona.io/", headers={"Proxy-Authorization": auth}
    )
    with opener.open(req, timeout=20) as response:
        assert response.status == 200
        write("proxy-login-check.json", {"status": response.status, **ready})
    with (ROOT / "proxychains.conf").open("x") as f:
        f.write(
            "strict_chain\nproxy_dns\nquiet_mode\ntcp_read_time_out 20000\ntcp_connect_time_out 15000\n"
        )
        f.write(
            "localnet 127.0.0.0/255.0.0.0\nlocalnet 10.0.0.0/255.0.0.0\n[ProxyList]\n"
        )
        f.write(f"http {ready['host']} {ready['port']} smoke {private['password']}\n")
    print(json.dumps({"login_https": 200, **ready}), flush=True)


try:
    {"start": start, "check": check}[sys.argv[1]]()
except BaseException as exc:
    print(json.dumps({"error_type": type(exc).__name__}), flush=True)
    raise SystemExit(1)
