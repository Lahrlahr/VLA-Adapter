import logging
import socket

import draccus
from dataclasses import dataclass
import websocket_policy_server

from vla_policy import VlaAdapter

@dataclass
class PolicyArgs():
    model_path : str = ''
    log_path : str = ''
    port : int = 2251

@draccus.wrap()
def main(args : PolicyArgs):
    policy = VlaAdapter(args.model_path, args.log_path)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    main()
