"""Short-lived, authenticated, internal-only CONNECT proxy for one smoke job."""

import base64
import hmac
import ipaddress
import json
import select
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from lifecycle import proxy_window


def destination(host, port):
    host = host.lower().rstrip(".")
    if port != 443 or not any(
        host == d or host.endswith("." + d) for d in ("daytona.io", "daytona.works")
    ):
        raise ValueError("Destination outside smoke allowlist")
    addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(a[4][0]).is_global for a in addresses
    ):
        raise ValueError("Non-public destination")
    return addresses[0][4]


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        with self.server.slots:
            self.request.settimeout(10)
            established = False
            try:
                raw = self.rfile.readline(8193)
                if len(raw) > 8192:
                    raise ValueError("Header too large")
                method, target, version = raw.decode("ascii").strip().split()
                headers, size = {}, len(raw)
                while True:
                    line = self.rfile.readline(8193)
                    size += len(line)
                    if size > 8192 or not line:
                        raise ValueError("Invalid headers")
                    if line == b"\r\n":
                        break
                    key, val = line.decode("ascii").split(":", 1)
                    headers[key.lower()] = val.strip()
                if not hmac.compare_digest(
                    headers.get("proxy-authorization", ""), self.server.auth
                ):
                    self.wfile.write(
                        b'HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm="smoke"\r\nContent-Length: 0\r\n\r\n'
                    )
                    return
                if method != "CONNECT" or version not in ("HTTP/1.0", "HTTP/1.1"):
                    raise ValueError("CONNECT required")
                host, port = target.rsplit(":", 1)
                address = destination(host, int(port))
                with socket.create_connection(address, timeout=10) as upstream:
                    self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    self.wfile.flush()
                    established = True
                    print(json.dumps({"connected_host": host}), flush=True)
                    transferred = 0
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        ready, _, _ = select.select(
                            [self.request, upstream], [], [], 30
                        )
                        if not ready:
                            break
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (
                                upstream if source is self.request else self.request
                            ).sendall(data)
                            transferred += len(data)
                    print(json.dumps({"host": host, "bytes": transferred}), flush=True)
            except Exception as exc:
                print(json.dumps({"error_type": type(exc).__name__}), flush=True)
                if not established:
                    try:
                        self.wfile.write(
                            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"
                        )
                    except OSError:
                        pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def process_request(self, request, address):
        if not self.workers.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.workers.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.workers.release()


def main(folder):
    folder = Path(folder)
    config = json.loads((folder / "private.json").read_text())
    address = config["bind"]
    if ipaddress.ip_address(address) not in ipaddress.ip_network("10.128.0.0/16"):
        raise ValueError("Must bind internal interface")
    with Server((address, 0), Handler) as server:
        server.slots = threading.BoundedSemaphore(16)
        server.workers = threading.BoundedSemaphore(16)
        server.auth = (
            "Basic "
            + base64.b64encode(("smoke:" + config["password"]).encode()).decode()
        )
        server.timeout = 1
        window = proxy_window(time.time())
        expires = window["expires_epoch"]
        with (folder / "proxy-ready.json").open("x") as f:
            json.dump({"host": address, "port": server.server_address[1], **window}, f)
        while time.time() < expires and not (folder / "stop-proxy").exists():
            server.handle_request()


if __name__ == "__main__":
    main(sys.argv[1])
