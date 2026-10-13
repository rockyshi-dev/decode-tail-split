"""Ownership checks: listening sockets, process sessions, and the physical GPU (by UUID)."""
import csv
import errno
import io
import os
from pathlib import Path
import socket
import subprocess
import time

from .telemetry import proc_fields, query_gpu


def processes(proc=Path('/proc')):
    result = {}
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = proc_fields(path / 'stat')
            result[int(path.name)] = {'state': fields[0], 'parent': int(fields[1]),
                                     'session': int(fields[3]), 'start': int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def owned_processes(pid, session, table=None):
    table = processes() if table is None else table
    owned = {pid} | {p for p, row in table.items() if row['session'] == session}
    while True:
        more = {p for p, row in table.items() if row['parent'] in owned}
        if more <= owned:
            break
        owned |= more
    return {p: table[p] for p in owned if p in table and table[p]['state'] not in ('Z', 'X')}


def precheck_port(port):
    # Keep both probes open until both succeed. IPv6 is optional.
    sockets = []
    try:
        for family, address in ((socket.AF_INET, '0.0.0.0'), (socket.AF_INET6, '::')):
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.bind((address, port))
            except OSError as error:
                if family != socket.AF_INET6 or error.errno != errno.EAFNOSUPPORT:
                    raise
    finally:
        for sock in sockets:
            sock.close()


def listener_ownership(port, owned, proc=Path('/proc')):
    listeners = set()
    for filename in ('tcp', 'tcp6'):
        path = proc / 'net' / filename
        if not path.exists() and filename == 'tcp6':
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                listeners.add(fields[9])
    owners = {}
    for pid in owned:
        try:
            for fd in (proc / str(pid) / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if target.startswith('socket:[') and target.endswith(']'):
                    owners.setdefault(target[8:-1], []).append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not listeners or not listeners <= owners.keys():
        raise RuntimeError('Listening socket does not belong to the owned server tree')
    return {inode: sorted(owners[inode]) for inode in sorted(listeners)}


def compute_apps():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True, timeout=5)
    return [{'pid': int(pid.strip()), 'uuid': uuid.strip()}
            for pid, uuid in csv.reader(io.StringIO(output))]


def gpu_ownership(owned, expected_uuid, gpu):
    """Our processes, and only ours, are the compute processes of GPU `gpu`, whose UUID is the
    one recorded before launch; none of our processes runs on another card."""
    state = query_gpu(gpu)
    apps = compute_apps()  # All cards: a worker on a different UUID must fail.
    ours = [row for row in apps if row['pid'] in owned]
    on_card = [row for row in apps if row['uuid'] == expected_uuid]
    if (state['uuid'] != expected_uuid or not ours or not on_card
        or not state['compute_pids'] or not set(state['compute_pids']) <= set(owned)
        or any(row['uuid'] != expected_uuid for row in ours)
        or any(row['pid'] not in owned for row in on_card)
        or {row['pid'] for row in on_card} != set(state['compute_pids'])):
        raise RuntimeError(f'Owned compute processes must occupy exactly GPU {gpu} ({expected_uuid})')
    return {'index': gpu, 'uuid': expected_uuid, 'state': state, 'owned_compute_apps': ours}


def wait_released(pid, session, expected_uuid, gpu, known=None, timeout=60):
    """Wait until every process of the session (and every known descendant) has exited and the
    GPU has no compute process and less than 100 MiB in use."""
    deadline = time.monotonic() + timeout
    known = dict(known or {})
    while True:
        table = processes()
        owned = owned_processes(pid, session, table)
        known.update({p: row['start'] for p, row in owned.items()})
        # Remember escaped descendants as well as reparented session members.
        alive = {p for p, start in known.items() if p in table
                 and table[p]['start'] == start and table[p]['state'] not in ('Z', 'X')}
        state = query_gpu(gpu)
        if state['uuid'] != expected_uuid:
            raise RuntimeError(f'GPU {gpu} UUID changed while waiting for release')
        if not alive and not state['compute_pids'] and state['memory_mib'] < 100:
            return {'gpu': state, 'live_pids': [], 'known_processes': known}
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Previous session or GPU memory did not release within {timeout}s: {sorted(alive)}')
        time.sleep(.25)
